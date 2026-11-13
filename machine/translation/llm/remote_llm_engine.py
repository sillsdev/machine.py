from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from html import escape
from math import exp
from time import sleep
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ...annotations.range import Range
from ...tokenization.latin_word_detokenizer import LatinWordDetokenizer
from ...tokenization.latin_word_tokenizer import LatinWordTokenizer
from ..translation_engine import TranslationEngine
from ..translation_result import TranslationResult
from ..translation_result_builder import TranslationResultBuilder
from ..translation_sources import TranslationSources
from ..word_alignment_matrix import WordAlignmentMatrix
from .example import Example
from .example_retriever import ExampleRetriever
from .llm_completion_client import Completion, LlmCompletionClient, LlmUsageTotals
from .openrouter_llm_completion_client import ContentBlockError, RefusalError

logger = logging.getLogger(__name__)


class RemoteLlmEngine(TranslationEngine):
    _GUIDANCE = (
        "Use the examples as evidence for {trg_lang} vocabulary, grammar, inflection, names, spelling and "
        "punctuation. Retain shared words and established borrowings when the examples support them; use "
        "natural target constructions instead of mechanically copying source wording.\n"
        "Preserve the full meaning of the source: participants, actions, negation, relationships and "
        "emphasis. Adapt an example's wording to the current sentence; similar examples may describe "
        "different people or events."
    )
    _SYSTEM_MESSAGE = (
        "Translate the final {src_lang} Bible verse into {trg_lang}, following the translation conventions "
        "demonstrated by this project's parallel examples.\n"
        + _GUIDANCE
        + " Nearby translations provide context, not additional content to translate.\n"
        "Return only the {trg_lang} translation of the final source verse on one line, without a label, "
        "verse number, explanation or alternative translations."
    )
    _BATCH_SYSTEM_MESSAGE = (
        "Translate the numbered {src_lang} Bible passages at the end into {trg_lang}, following the "
        "translation conventions demonstrated by this project's parallel examples.\n"
        + _GUIDANCE
        + "\nReturn only the {trg_lang} translations, one numbered line per source passage, without labels, "
        "explanations or alternative translations."
    )
    _SINGLE_TASK = (
        "only this {src_lang} verse into {trg_lang}.\n<source_to_translate>\n{source}\n</source_to_translate>\n"
        "{trg_lang}:"
    )
    _BATCH_TASK = (
        "each of these {num_segments} consecutive {src_lang} passages into {trg_lang}. Some may be section "
        "headings rather than verses. Read them together, so that participants, pronouns, and the flow of the "
        "passage stay consistent across them, but translate each one on its own.\n"
        "Reply with exactly {num_segments} lines, one per passage, in the same order, each formatted as "
        "`<number>. <translation>`. Do not merge, split, reorder, or omit passages, and do not add any other "
        "text.\n<source_to_translate>\n{source}\n</source_to_translate>"
    )
    _BACK_TRANSLATION_GUIDANCE = (
        "A back-translation shows what the {src_lang} text ACTUALLY says, so a reviewer who does not read "
        "{src_lang} can check it. Render the {src_lang} wording faithfully, including anything unexpected, "
        "unusual, added, missing or apparently mistaken \u2014 do NOT smooth it toward a familiar or expected wording, "
        "and do not supply content that is not in the {src_lang}."
    )
    _MAX_COMPLETION_ATTEMPTS = 3
    _RETRY_DELAY_SECONDS = 0.5
    _NUMBERED_LINE = re.compile(r"^\s*(\d{1,4})\s*[.):\]]\s*(.*)$")
    _CODE_FENCE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?\s*```\s*$", re.DOTALL)
    _LABELS = ("target", "translation", "output", "answer")

    def __init__(
        self,
        completion_client: LlmCompletionClient,
        example_retriever: ExampleRetriever,
        source_language: str,
        target_language: str,
        num_examples: int = 100,
        batch_size: int = 15,
        num_workers: int = 4,
        usage_totals: Optional[LlmUsageTotals] = None,
        prompt: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if num_examples < 0:
            raise ValueError("num_examples must be non-negative.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        if num_workers < 1:
            raise ValueError("num_workers must be at least one.")
        if not source_language.strip() or not target_language.strip():
            raise ValueError("Source and target languages must be specified.")
        self._completion_client = completion_client
        self._example_retriever = example_retriever
        self._source_language = source_language
        self._target_language = target_language
        self._num_examples = num_examples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._prompt = dict(prompt or {})
        self._tokenizer = LatinWordTokenizer()
        self._detokenizer = LatinWordDetokenizer()
        self._target_trigrams = {
            target[i : i + 3]
            for example in self._example_retriever.examples
            for target in [example.target.strip()]
            for i in range(len(target) - 2)
        }
        self.usage_totals = usage_totals if usage_totals is not None else LlmUsageTotals()

    def translate(self, segment: Union[str, Sequence[str]]) -> TranslationResult:
        return self.translate_batch([segment])[0]

    def translate_n(self, n: int, segment: Union[str, Sequence[str]]) -> Sequence[TranslationResult]:
        return self.translate_n_batch(n, [segment])[0]

    def translate_batch(self, segments: Sequence[Union[str, Sequence[str]]]) -> Sequence[TranslationResult]:
        if isinstance(segments, str):
            segments = [segments]
        results: List[TranslationResult] = []
        segment_list = list(segments)
        batches = [segment_list[i : i + self._batch_size] for i in range(0, len(segment_list), self._batch_size)]
        if self._num_workers == 1 or len(batches) <= 1:
            batch_results = [self._complete_batch(segment_batch) for segment_batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                batch_results = list(executor.map(self._complete_batch, batches))
        for batch_result in batch_results:
            results.extend(batch_result)
        return results

    def translate_n_batch(
        self, n: int, segments: Sequence[Union[str, Sequence[str]]]
    ) -> Sequence[Sequence[TranslationResult]]:
        if isinstance(segments, str):
            segments = [segments]
        if n <= 0:
            return [[] for _ in segments]
        results_by_draft = [self.translate_batch(segments) for _ in range(n)]
        return [list(results) for results in zip(*results_by_draft, strict=True)]

    def _complete_batch(self, segments: Sequence[Union[str, Sequence[str]]]) -> List[TranslationResult]:
        normalized = [self._normalize_segment(segment) for segment in segments]
        nonempty_indices = [i for i, (text, _) in enumerate(normalized) if text.strip()]
        results: List[Optional[TranslationResult]] = [None] * len(segments)
        for index, (text, tokens) in enumerate(normalized):
            # If there's no text to translate, do not request LLM completion
            if not text.strip():
                results[index] = self._to_result(tokens, Completion(""))
        if not nonempty_indices:
            return [result for result in results if result is not None]

        nonempty_sources = [normalized[i][0] for i in nonempty_indices]
        errors: Dict[int, str] = {}
        try:
            if len(nonempty_sources) == 1:
                translations = [self._complete_single(nonempty_sources[0])]
            else:
                translations = self._complete_numbered(nonempty_sources)
        except (ContentBlockError, RefusalError) as error:
            if len(nonempty_sources) == 1:
                translations = [Completion("")]
                errors[nonempty_indices[0]] = str(error)
            else:
                logger.warning("Batch completion failed after retries; retrying its segments individually.")
                translations = []
                for index, source in zip(nonempty_indices, nonempty_sources, strict=True):
                    try:
                        translations.append(self._complete_single(source))
                    except (ContentBlockError, RefusalError) as error:
                        translations.append(Completion(""))
                        errors[index] = str(error)
        for index, translation in zip(nonempty_indices, translations, strict=True):
            _, tokens = normalized[index]
            results[index] = self._to_result(tokens, translation, errors.get(index))
        return [result for result in results if result is not None]

    def _complete_numbered(self, sources: Sequence[str]) -> List[Completion]:
        messages = self._build_messages(sources)
        first = self._request(messages)
        parsed = self._sanitize_numbered_response(first.text, len(sources))
        if parsed is None:
            correction_messages = messages + [
                {"role": "assistant", "content": first.text},
                {
                    "role": "user",
                    "content": (
                        f"That reply did not have the required format. Reply again with exactly "
                        f"{len(sources)} lines, one per segment, in the same order, each formatted as "
                        "`<number>. <translation>`, and nothing else."
                    ),
                },
            ]
            corrected = self._request(correction_messages)
            parsed = self._sanitize_numbered_response(corrected.text, len(sources))
        if parsed is not None:
            return [Completion(text) for text in parsed]

        logger.warning("Could not parse %d translations; splitting the request and retrying.", len(sources))
        middle = len(sources) // 2
        return self._complete_numbered_or_single(sources[:middle]) + self._complete_numbered_or_single(sources[middle:])

    def _complete_numbered_or_single(self, sources: Sequence[str]) -> List[Completion]:
        if len(sources) == 1:
            return [self._complete_single(sources[0])]
        return self._complete_numbered(sources)

    def _complete_single(self, source: str) -> Completion:
        messages = self._build_messages([source])
        raw_completion = self._request(messages, logprobs=self._completion_client.supports_logprobs())
        translation = self._sanitize_single_response(raw_completion.text)
        if not translation:
            corrected = self._request(
                messages,
                logprobs=self._completion_client.supports_logprobs(),
            )
            raw_completion = corrected
            translation = self._sanitize_single_response(raw_completion.text)
        if translation != raw_completion.text:
            return Completion(
                translation,
                cost=raw_completion.cost,
                input_token_count=raw_completion.input_token_count,
                output_token_count=raw_completion.output_token_count,
            )
        return raw_completion

    def _request(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        for attempt in range(self._MAX_COMPLETION_ATTEMPTS):
            try:
                completion = self._completion_client.complete(messages, logprobs=logprobs)
            except (ContentBlockError, RefusalError):
                if attempt + 1 == self._MAX_COMPLETION_ATTEMPTS:
                    raise
                sleep(0.5 * (2**attempt))
            else:
                self.usage_totals.add(completion)
                return completion
        raise RuntimeError("Completion attempts ended unexpectedly.")

    def _build_messages(self, sources: Sequence[str]) -> List[Dict[str, str]]:
        query = "\n".join(sources)
        pool = self._example_retriever.examples
        whole_pool = self._num_examples > 0 and self._num_examples >= len(pool)
        examples = pool if whole_pool else list(reversed(self._example_retriever.rank(query, self._num_examples)))
        rendered_examples = self._format_examples(examples)
        corpus_block = (
            f"<translation_examples>\n{rendered_examples}\n</translation_examples>"
            if (whole_pool and rendered_examples)
            else ""
        )
        batch = len(sources) > 1
        system = self._prompt.get("system_message")
        if batch and self._prompt.get("batch_system_message") is not None:
            system = self._prompt["batch_system_message"]
        if system is None:
            system = self._BATCH_SYSTEM_MESSAGE if batch else self._SYSTEM_MESSAGE
        if self._prompt.get("back_translation") and self._BACK_TRANSLATION_GUIDANCE not in system:
            system = f"{system}\n{self._BACK_TRANSLATION_GUIDANCE}" if system else self._BACK_TRANSLATION_GUIDANCE
        task = self._BATCH_TASK if batch else self._SINGLE_TASK
        instruction = self._prompt.get("batch_instruction_template" if batch else "instruction_template")
        if instruction is None or (corpus_block and not instruction):
            instruction = (
                "<translation_examples>\n{examples}\n</translation_examples>\n\n"
                "Following the project's examples above, translate " + task
                if self._num_examples > 0 and not corpus_block
                else "Translate " + task
            )
        source = "\n".join(f"{index}. {text}" for index, text in enumerate(sources, 1)) if batch else sources[0]
        format_options = self._example_format()
        escape_source = format_options["type"] == "text" and format_options.get("escape", False)
        values = {
            "src_lang": self._source_language,
            "trg_lang": self._target_language,
            "source": escape(source, quote=False) if escape_source else source,
            "examples": "" if corpus_block else rendered_examples,
            "num_segments": len(sources),
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system.format(**values)})
        if corpus_block:
            if messages:
                messages[0]["content"] += f"\n\n{corpus_block}"
            else:
                messages.append({"role": "system", "content": corpus_block})
        messages.append({"role": "user", "content": instruction.format(**values)})
        return messages

    def _example_format(self) -> Dict[str, Any]:
        options = self._prompt.get("example_format")
        if options is None:
            return {
                "type": "text",
                "template": "{src_lang}: {source}\n{trg_lang}: {target}",
                "separator": "\n\n",
                "escape": True,
            }
        if isinstance(options, str):
            return {"type": options}
        return {"type": "text", **options}

    def _format_examples(self, examples: Sequence[Example]) -> str:
        options = self._example_format()
        format_type = options["type"]
        if format_type == "json":
            return (
                json.dumps(
                    [{"source": ex.source, "target": ex.target} for ex in examples], ensure_ascii=False, indent=2
                )
                + "\n"
                if examples
                else ""
            )
        if format_type == "xml":
            return "".join(
                f"<example>\n<source>{escape(ex.source, quote=False)}</source>\n"
                f"<target>{escape(ex.target, quote=False)}</target>\n</example>\n"
                for ex in examples
            )
        if format_type != "text":
            raise ValueError(f"Unknown example_format.type '{format_type}'.")
        template = options.get("template")
        if template is None:
            template = "Source ({src_lang}): {source}\nTranslation ({trg_lang}): {target}\n\n"
        separator = options.get("separator") or ""
        return separator.join(
            template.format(
                src_lang=self._source_language,
                trg_lang=self._target_language,
                source=escape(ex.source, quote=False) if options.get("escape") else ex.source,
                target=escape(ex.target, quote=False) if options.get("escape") else ex.target,
            )
            for ex in examples
        )

    def _sanitize_numbered_response(self, text: str, expected_count: int) -> Optional[List[str]]:
        match = self._CODE_FENCE.match(text.strip())
        if match is not None:
            text = match.group(1)
        parsed: Dict[int, List[str]] = {}
        current: Optional[int] = None
        for line in text.splitlines():
            match = self._NUMBERED_LINE.match(line)
            if match is not None:
                index = int(match.group(1))
                if index in parsed:
                    return None
                current = index
                parsed[index] = [match.group(2).strip()]
            elif current is not None and line.strip():
                parsed[current].append(line.strip())
        if set(parsed) != set(range(1, expected_count + 1)):
            return None
        translations = [" ".join(part for part in parsed[i] if part).strip() for i in range(1, expected_count + 1)]
        return None if any(not translation for translation in translations) else translations

    def _sanitize_single_response(self, text: str) -> str:
        match = self._CODE_FENCE.match(text.strip())
        if match is not None:
            text = match.group(1)
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip().strip("`").strip()
            label_pattern = "|".join(re.escape(label) for label in (self._target_language, *self._LABELS) if label)
            if label_pattern:
                line = re.sub(rf"^({label_pattern})\s*[:\-]\s*", "", line, flags=re.IGNORECASE).strip()
            if line:
                lines.append(line)
        if not lines:
            return ""
        return max(
            enumerate(lines),
            key=lambda item: (
                sum(1 for i in range(len(item[1]) - 2) if item[1][i : i + 3] in self._target_trigrams)
                / max(len(item[1]) - 2, 1),
                item[0],
            ),
        )[1]

    def _normalize_segment(self, segment: Union[str, Sequence[str]]) -> Tuple[str, List[str]]:
        if isinstance(segment, str):
            return segment, list(self._tokenizer.tokenize(segment))
        tokens = list(segment)
        return self._detokenizer.detokenize(tokens), tokens

    def _to_result(
        self, source_tokens: List[str], completion: Completion, error: Optional[str] = None
    ) -> TranslationResult:
        translation = completion.text.strip()
        target_tokens = list(self._tokenizer.tokenize(translation))
        builder = TranslationResultBuilder(source_tokens)
        for token in target_tokens:
            builder.append_token(token, TranslationSources.LLM, -1.0)
        builder.mark_phrase(
            Range.create(0, len(source_tokens)),
            WordAlignmentMatrix.from_word_pairs(len(source_tokens), len(target_tokens)),
        )
        if completion.token_logprobs and translation == completion.text:
            mean_logprob = sum(token.logprob for token in completion.token_logprobs) / len(completion.token_logprobs)
            builder.set_sequence_confidence(exp(min(0.0, mean_logprob)))
        return builder.to_result(translation, error)

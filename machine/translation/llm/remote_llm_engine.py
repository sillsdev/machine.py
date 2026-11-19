from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from math import exp
from time import sleep
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ...annotations.range import Range
from ...tokenization.latin_word_detokenizer import LatinWordDetokenizer
from ...tokenization.latin_word_tokenizer import LatinWordTokenizer
from ..translation_engine import TranslationEngine
from ..translation_result import TranslationResult
from ..translation_result_builder import TranslationResultBuilder
from ..translation_sources import TranslationSources
from ..word_alignment_matrix import WordAlignmentMatrix
from .example_retriever import ExampleRetriever
from .llm_completion_client import Completion, LlmCompletionClient, LlmUsageTotals
from .openrouter_llm_completion_client import ContentBlockError, RefusalError

logger = logging.getLogger(__name__)


class RemoteLlmEngine(TranslationEngine):
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
        self._tokenizer = LatinWordTokenizer()
        self._detokenizer = LatinWordDetokenizer()
        self._target_trigrams = {
            target[i : i + 3]
            for example in self._example_retriever.examples
            for target in [example.target.strip()]
            for i in range(len(target) - 2)
        }
        self.usage_totals = usage_totals if usage_totals is not None else LlmUsageTotals()
        self.failed_segment_indices: List[int] = []

    def translate(self, segment: Union[str, Sequence[str]]) -> TranslationResult:
        return self.translate_batch([segment])[0]

    def translate_n(self, n: int, segment: Union[str, Sequence[str]]) -> Sequence[TranslationResult]:
        return self.translate_n_batch(n, [segment])[0]

    def translate_batch(self, segments: Sequence[Union[str, Sequence[str]]]) -> Sequence[TranslationResult]:
        if isinstance(segments, str):
            segments = [segments]
        results: List[TranslationResult] = []
        self.failed_segment_indices = []
        segment_list = list(segments)
        batches = [segment_list[i : i + self._batch_size] for i in range(0, len(segment_list), self._batch_size)]
        if self._num_workers == 1 or len(batches) <= 1:
            batch_results = [self._complete_batch(segment_batch) for segment_batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                batch_results = list(executor.map(self._complete_batch, batches))
        offset = 0
        for batch_result, failed_indices in batch_results:
            results.extend(batch_result)
            self.failed_segment_indices.extend(offset + index for index in failed_indices)
            offset += len(batch_result)
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

    def _complete_batch(
        self, segments: Sequence[Union[str, Sequence[str]]]
    ) -> Tuple[List[TranslationResult], List[int]]:
        normalized = [self._normalize_segment(segment) for segment in segments]
        nonempty_indices = [i for i, (text, _) in enumerate(normalized) if text.strip()]
        results: List[Optional[TranslationResult]] = [None] * len(segments)
        for index, (text, tokens) in enumerate(normalized):
            # If there's no text to translate, do not request LLM completion
            if not text.strip():
                results[index] = self._to_result(tokens, Completion(""))
        if not nonempty_indices:
            return [result for result in results if result is not None], []

        nonempty_sources = [normalized[i][0] for i in nonempty_indices]
        failed_indices: List[int] = []
        try:
            if len(nonempty_sources) == 1:
                translations = [self._complete_single(nonempty_sources[0])]
            else:
                translations = self._complete_numbered(nonempty_sources)
        except (ContentBlockError, RefusalError):
            if len(nonempty_sources) == 1:
                translations = [Completion("")]
                failed_indices.append(nonempty_indices[0])
            else:
                logger.warning("Batch completion failed after retries; retrying its segments individually.")
                translations = []
                for index, source in zip(nonempty_indices, nonempty_sources, strict=True):
                    try:
                        translations.append(self._complete_single(source))
                    except (ContentBlockError, RefusalError):
                        translations.append(Completion(""))
                        failed_indices.append(index)
        for index, translation in zip(nonempty_indices, translations, strict=True):
            _, tokens = normalized[index]
            results[index] = self._to_result(tokens, translation)
        return [result for result in results if result is not None], failed_indices

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
                        f"That reply was not in the required format. Reply with exactly {len(sources)} lines, "
                        "one translation per source in the same order, formatted as `<number>. <translation>`, "
                        "and no other text."
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
                messages
                + [
                    {"role": "assistant", "content": raw_completion.text},
                    {
                        "role": "user",
                        "content": (
                            f"Reply with only the {self._target_language} translation of the source passage. "
                            "Do not include a label, explanation, or formatting."
                        ),
                    },
                ],
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
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"Translate from {self._source_language} to {self._target_language}. "
                    "Use the examples as guidance. Return only the translation, with no explanation."
                ),
            }
        ]
        for example in self._example_retriever.rank(query, self._num_examples):
            messages.append({"role": "user", "content": example.source})
            messages.append({"role": "assistant", "content": example.target})
        if len(sources) == 1:
            messages.append({"role": "user", "content": sources[0]})
        else:
            numbered_sources = "\n".join(f"{i}. {source}" for i, source in enumerate(sources, 1))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Translate each of these {len(sources)} passages into {self._target_language}. "
                        "Return exactly one numbered line per passage in the same order, with no other text.\n"
                        f"{numbered_sources}"
                    ),
                }
            )
        return messages

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

    def _to_result(self, source_tokens: List[str], completion: Completion) -> TranslationResult:
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
        return builder.to_result(translation)

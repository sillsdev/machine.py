import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape, quoteattr, unescape

from ...utils.progress_status import ProgressStatus
from .completion_finish_reason import CompletionFinishReason
from .context_verse import ContextVerse
from .example import Example
from .example_retriever import ExampleRetriever
from .llm_completion_client import (
    CompletionError,
    ContentBlockError,
    LlmCompletionClient,
    MalformedCompletionError,
    RefusalError,
    TruncatedCompletionError,
)
from .post_edit_batch_error import PostEditBatchError
from .post_edit_result import PostEditResult
from .post_edit_segment import PostEditSegment

logger = logging.getLogger(__name__)

DEFAULT_NUM_EXAMPLES = 50
DEFAULT_MAX_ATTEMPTS = 3

_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are an expert translator and post-editor from {source_language} to {target_language}.

Three blocks, all selected for this one verse:

- <translation_memory>: {source_language} → {target_language} examples of verses with
  similar wording. Use them as translation support — follow their vocabulary,
  spelling of names, and style closely.
- <passage_context>: the verses around this one, labelled by offset (offset="-1"
  is the verse before), with their {target_language} where a translation exists.
  Background only: NEVER translate them, NEVER put them in your output.
- <draft_translation>: an unreviewed machine translation of this verse.
  status="not_available" means there is none — translate from scratch instead.

YOUR JOB. You are post-editing the draft translation to improve it by fixing clear
errors and omissions{justification_task}. Read the translation
memory, passage context, and source text; use them as context for post-editing. The
draft is a dynamic translation that has learned the translator's style, so it's
intentionally not a literal translation; don't post-edit it to be one, and go with the
translator's style over the source's style. Favor making more fixes rather than less,
but without mistakes.

Translate the verse marked role="to_translate" and nothing else.

NEVER write the "translation" in any language other than {target_language}.

OUTPUT — one JSON object:
{justification_fields}- "translation": your {target_language} text for the verse marked role="to_translate"."""

_PASSAGE_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are an expert translator and post-editor from {source_language} to {target_language}.

Three blocks, all selected for these verses:

- <translation_memory>: {source_language} → {target_language} examples of verses with
  similar wording. Use them as translation support — follow their vocabulary,
  spelling of names, and style closely.
{passage_context}
- <draft_translation>: an unreviewed machine translation of each numbered verse.
  status="not_available" means there is none for that verse — translate it from
  scratch instead.

YOUR JOB. You are post-editing the draft translations to improve them by fixing clear
errors and omissions{justification_task}. Read the translation
memory, passage context, and source text; use them as context for post-editing. The
drafts are a dynamic translation that has learned the translator's style, so they are
intentionally not a literal translation; don't post-edit them to be one, and go with the
translator's style over the source's style. Favor making more fixes rather than less,
but without mistakes.

{reading_guidance}

NEVER write a "translation" in any language other than {target_language}.

OUTPUT — one JSON object whose "verses" list has one entry per numbered verse, in order:
- "number": the verse's number.
{justification_fields}- "translation": your {target_language} text for that verse."""

_CONSECUTIVE_PASSAGE_CONTEXT = """\
- <passage_context>: the passage, labelled by offset from its first numbered verse.
  The numbered verses marked role="to_translate" are the ones to post-edit. The other
  verses, with their {target_language} where a translation exists, are background only:
  NEVER translate them, NEVER put them in your output."""

_CONSECUTIVE_READING_GUIDANCE = """\
Read the numbered verses together, so that participants, pronouns, and the flow of the
passage stay consistent across them, but post-edit each one on its own. Do not merge,
split, reorder, or omit verses."""

_UNRELATED_PASSAGE_CONTEXT = """\
- <passage_context>: one for each numbered verse, which is marked role="to_translate",
  with the verses around it labelled by offset (offset="-1" is the verse before). The
  other verses, with their {target_language} where a translation exists, are background
  only: NEVER translate them, NEVER put them in your output."""

_UNRELATED_READING_GUIDANCE = """\
The numbered verses come from unrelated parts of the text. Post-edit each one on its
own, using only its own passage context. Do not merge, split, reorder, or omit verses."""

_JUSTIFICATION_TASK = " and providing your justification for the fix"

_JUSTIFICATION_FIELDS = """- "changed": true if your text differs from the draft. false otherwise.
- "justification": if your text differs from the draft, one sentence in English — what
  you fixed and why. Otherwise an empty string.
"""

_RETRY_REMINDER_TEMPLATE = (
    "\n\nReply with a single JSON object and nothing else (no markdown fence), with exactly the keys {keys}, "
    'and a non-empty "translation".'
)

_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
}

_JUSTIFIED_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "justification": {"type": "string"},
        "translation": {"type": "string"},
    },
    "required": ["changed", "justification", "translation"],
}


def _create_passage_response_schema(verse_schema: Dict[str, Any]) -> Dict[str, Any]:
    verse = {
        "type": "object",
        "properties": {"number": {"type": "integer"}, **verse_schema["properties"]},
        "required": ["number", *verse_schema["required"]],
        # Strict structured output requires this on every object, and the completion client sets it only at the top.
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"verses": {"type": "array", "items": verse}},
        "required": ["verses"],
    }


def _to_single_line(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())
    return " ".join(line for line in lines if line != "")


def _format_examples(examples: Sequence[Example], source_language: str, target_language: str) -> str:
    source_lang = quoteattr(source_language)
    target_lang = quoteattr(target_language)
    lines = ["<translation_memory>"]
    for example in examples:
        lines.append(
            f"  <example>\n"
            f"    <source lang={source_lang}>{escape(example.source.strip())}</source>\n"
            f"    <target lang={target_lang}>{escape(example.target.strip())}</target>\n"
            f"  </example>"
        )
    lines.append("</translation_memory>")
    return "\n".join(lines)


def _format_passage(
    members: Sequence[Tuple[int, PostEditSegment]],
    source_language: str,
    target_language: str,
    first_number: Optional[int] = None,
) -> str:
    """Each member is a segment with its position in the passage. Given first_number, they are numbered from it."""
    first_position = members[0][0]
    to_translate = {position - first_position: segment for position, segment in members}
    numbers = (
        {offset: number for number, offset in enumerate(to_translate, first_number)} if first_number is not None else {}
    )
    context_by_offset: Dict[int, ContextVerse] = {}
    for position, segment in members:
        for verse in segment.context_verses:
            offset = position - first_position + verse.offset_from_segment
            if verse.source.strip() and offset not in to_translate:
                context_by_offset.setdefault(offset, verse)
    offsets = sorted([*context_by_offset, *to_translate])
    source_lang = quoteattr(source_language)
    target_lang = quoteattr(target_language)
    lines = ["<passage_context>"]
    for offset in range(offsets[0], offsets[-1] + 1):
        if offset in to_translate:
            number = f' number="{numbers[offset]}"' if offset in numbers else ""
            lines.append(
                f'  <verse offset="{offset}" role="to_translate"{number}>\n'
                f"    <source lang={source_lang}>{escape(to_translate[offset].source.strip())}</source>\n"
                f"  </verse>"
            )
        elif offset in context_by_offset:
            verse = context_by_offset[offset]
            target = verse.target.strip()
            rendered_target = (
                f"<target lang={target_lang}>{escape(target)}</target>"
                if target
                else f'<target lang={target_lang} status="not_available"/>'
            )
            lines.append(
                f'  <verse offset="{offset}">\n'
                f"    <source lang={source_lang}>{escape(verse.source.strip())}</source>\n"
                f"    {rendered_target}\n"
                f"  </verse>"
            )
        else:
            lines.append(f'  <missing_verse offset="{offset}"/>')
    lines.append("</passage_context>")
    return "\n".join(lines)


def _format_draft(draft: str, target_language: str) -> str:
    draft = draft.strip()
    target_lang = quoteattr(target_language)
    if not draft:
        return f'<draft_translation lang={target_lang} status="not_available"/>'
    return f"<draft_translation lang={target_lang}>\n  {escape(draft)}\n</draft_translation>"


def _format_numbered_drafts(drafts: Sequence[str], target_language: str) -> str:
    lines = [f"<draft_translation lang={quoteattr(target_language)}>"]
    for number, draft in enumerate(drafts, 1):
        draft = draft.strip()
        lines.append(
            f'  <draft number="{number}">{escape(draft)}</draft>'
            if draft
            else f'  <draft number="{number}" status="not_available"/>'
        )
    lines.append("</draft_translation>")
    return "\n".join(lines)


def _try_parse_passage_response(
    text: str, drafts: Sequence[str], include_justification: bool
) -> Optional[List[PostEditResult]]:
    """Returns None unless the reply has a usable translation for every verse, numbered in order."""
    verses = json.loads(text)["verses"]
    if [verse["number"] for verse in verses] != list(range(1, len(drafts) + 1)):
        return None
    results: List[PostEditResult] = []
    for verse, draft in zip(verses, drafts):
        result = _parse_verse(verse, draft, include_justification)
        if result is None:
            return None
        results.append(result)
    return results


def _parse_verse(data: Dict[str, Any], draft: str, include_justification: bool) -> Optional[PostEditResult]:
    translation = unescape(_to_single_line(data["translation"]))
    if not translation:
        return None
    changed = translation != draft
    return PostEditResult(
        translation=translation,
        succeeded=True,
        changed=changed,
        justification=unescape(data["justification"].strip()) if changed and include_justification else "",
    )


def _split_evenly(total: int, count: int) -> List[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + 1 if index < remainder else quotient for index in range(count)]


def _with_shared_usage(
    results: Sequence[PostEditResult], cost: float, input_token_count: int, output_token_count: int
) -> List[PostEditResult]:
    count = len(results)
    return [
        replace(
            result,
            total_cost=result.total_cost + cost / count,
            total_input_token_count=result.total_input_token_count + input_share,
            total_output_token_count=result.total_output_token_count + output_share,
        )
        for result, input_share, output_share in zip(
            results, _split_evenly(input_token_count, count), _split_evenly(output_token_count, count)
        )
    ]


class LlmPostEditor:
    """The completion client is borrowed: the caller opens and closes it, and may share it between post-editors."""

    def __init__(
        self,
        completion_client: LlmCompletionClient,
        example_retriever: ExampleRetriever,
        source_language: str,
        target_language: str,
        num_examples: int = DEFAULT_NUM_EXAMPLES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        include_justification: bool = False,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self._completion_client = completion_client
        self._example_retriever = example_retriever
        self._source_language = source_language
        self._target_language = target_language
        self._num_examples = num_examples
        self._max_attempts = max_attempts
        self._include_justification = include_justification
        template_fields = {
            "source_language": source_language,
            "target_language": target_language,
            "justification_task": _JUSTIFICATION_TASK if include_justification else "",
            "justification_fields": _JUSTIFICATION_FIELDS if include_justification else "",
        }
        self._system_instruction = _SYSTEM_INSTRUCTION_TEMPLATE.format(**template_fields)
        self._passage_system_instructions = {
            consecutive: _PASSAGE_SYSTEM_INSTRUCTION_TEMPLATE.format(
                passage_context=passage_context.format(**template_fields),
                reading_guidance=reading_guidance,
                **template_fields,
            )
            for consecutive, passage_context, reading_guidance in [
                (True, _CONSECUTIVE_PASSAGE_CONTEXT, _CONSECUTIVE_READING_GUIDANCE),
                (False, _UNRELATED_PASSAGE_CONTEXT, _UNRELATED_READING_GUIDANCE),
            ]
        }
        self._response_schema = _JUSTIFIED_RESPONSE_SCHEMA if include_justification else _RESPONSE_SCHEMA
        self._passage_response_schema = _create_passage_response_schema(self._response_schema)
        self._retry_reminder = _RETRY_REMINDER_TEMPLATE.format(
            keys=", ".join(f'"{key}"' for key in self._response_schema["required"])
        )

    def post_edit(self, segment: PostEditSegment) -> PostEditResult:
        draft = _to_single_line(segment.draft)
        if not segment.source.strip():
            return PostEditResult(translation=draft)
        prompt = self._build_prompt(segment)
        total_cost = 0.0
        total_input_token_count = 0
        total_output_token_count = 0
        finish_reason = CompletionFinishReason.UNKNOWN
        for attempt in range(1, self._max_attempts + 1):
            messages = [
                {"role": "system", "content": self._system_instruction},
                {"role": "user", "content": prompt if attempt == 1 else prompt + self._retry_reminder},
            ]
            try:
                completion = self._completion_client.complete(messages, response_schema=self._response_schema)
            except (ContentBlockError, RefusalError, TruncatedCompletionError, MalformedCompletionError) as error:
                total_cost += error.cost
                total_input_token_count += error.input_token_count
                total_output_token_count += error.output_token_count
                if isinstance(error, MalformedCompletionError):
                    logger.warning("Post-edit attempt %d/%d failed: %s", attempt, self._max_attempts, error)
                    continue
                if isinstance(error, ContentBlockError):
                    finish_reason = CompletionFinishReason.CONTENT_FILTER
                elif isinstance(error, RefusalError):
                    finish_reason = CompletionFinishReason.REFUSAL
                else:
                    finish_reason = CompletionFinishReason.LENGTH
                logger.warning(
                    "Post-edit attempt %d/%d stopped without retrying: %s", attempt, self._max_attempts, error
                )
                break
            total_cost += completion.cost
            total_input_token_count += completion.input_token_count
            total_output_token_count += completion.output_token_count
            finish_reason = completion.finish_reason
            result = _parse_verse(json.loads(completion.text), draft, self._include_justification)
            if result is not None:
                return replace(
                    result,
                    total_cost=total_cost,
                    total_input_token_count=total_input_token_count,
                    total_output_token_count=total_output_token_count,
                    finish_reason=finish_reason,
                )
            logger.warning(
                "Post-edit attempt %d/%d produced no usable translation (%s).",
                attempt,
                self._max_attempts,
                finish_reason.name,
            )
        return PostEditResult(
            translation=draft,
            total_cost=total_cost,
            total_input_token_count=total_input_token_count,
            total_output_token_count=total_output_token_count,
            finish_reason=finish_reason,
        )

    def post_edit_batch(
        self,
        segments: Sequence[PostEditSegment],
        max_workers: int = 1,
        segments_per_request: int = 1,
        consecutive: bool = True,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
    ) -> Sequence[PostEditResult]:
        """Returns a result per segment, in order, or raises PostEditBatchError after all finish if any failed.

        The error holds a result for every segment, with the draft unchanged for each failed one, and each failure's
        exception by segment index.

        Segments are sent segments_per_request at a time, and a failed request fails all of its segments. Each result
        carries an even share of its request's cost and tokens. When consecutive, each request's segments must be
        consecutive verses in order and are shown as one passage; otherwise each is shown with its own context.

        Up to max_workers requests run at once. progress is called on the calling thread as segments finish.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if segments_per_request < 1:
            raise ValueError("segments_per_request must be at least 1.")
        runs = [
            segments[start : start + segments_per_request] for start in range(0, len(segments), segments_per_request)
        ]
        run_results: List[List[PostEditResult]] = [[] for _ in runs]
        errors: Dict[int, Exception] = {}
        finished_count = 0
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures: Dict[Future, int] = {
                executor.submit(self._post_edit_run, run, consecutive): run_index for run_index, run in enumerate(runs)
            }
            for future in as_completed(futures):
                run_index = futures[future]
                run = runs[run_index]
                try:
                    run_results[run_index] = future.result()
                except Exception as error:
                    first = run_index * segments_per_request
                    logger.error("Post-edit of segments %d-%d failed.", first, first + len(run) - 1, exc_info=error)
                    errors.update((index, error) for index in range(first, first + len(run)))
                    run_results[run_index] = [
                        PostEditResult(translation=_to_single_line(segment.draft)) for segment in run
                    ]
                finished_count += len(run)
                if progress is not None:
                    progress(ProgressStatus.from_step(finished_count, len(segments)))
        finally:
            # Stops queued runs from starting, and so from being billed, when the wait is interrupted.
            executor.shutdown(wait=True, cancel_futures=True)
        results = [result for run_result in run_results for result in run_result]
        if errors:
            errors = dict(sorted(errors.items()))
            raise PostEditBatchError(results, errors) from next(iter(errors.values()))
        return results

    def _post_edit_run(self, segments: Sequence[PostEditSegment], consecutive: bool) -> List[PostEditResult]:
        members = [(position, segment) for position, segment in enumerate(segments) if segment.source.strip()]
        edited = iter(self._post_edit_passage(members, consecutive) if members else [])
        return [
            next(edited) if segment.source.strip() else PostEditResult(translation=_to_single_line(segment.draft))
            for segment in segments
        ]

    def _post_edit_passage(
        self, members: Sequence[Tuple[int, PostEditSegment]], consecutive: bool
    ) -> List[PostEditResult]:
        """Each member is a segment with a source and its position in the run."""
        if len(members) == 1:
            return [self.post_edit(members[0][1])]
        drafts = [_to_single_line(segment.draft) for _, segment in members]
        messages = [
            {"role": "system", "content": self._passage_system_instructions[consecutive]},
            {"role": "user", "content": self._build_passage_prompt(members, consecutive)},
        ]
        results: Optional[List[PostEditResult]] = None
        try:
            completion = self._completion_client.complete(messages, response_schema=self._passage_response_schema)
        except CompletionError as error:
            logger.warning("Post-edit of %d verses in one request failed: %s", len(members), error)
            cost, input_token_count, output_token_count = error.cost, error.input_token_count, error.output_token_count
        else:
            cost = completion.cost
            input_token_count = completion.input_token_count
            output_token_count = completion.output_token_count
            results = _try_parse_passage_response(completion.text, drafts, self._include_justification)
            if results is None:
                logger.warning("Post-edit of %d verses in one request produced no usable translations.", len(members))
            else:
                results = [replace(result, finish_reason=completion.finish_reason) for result in results]
        if results is None:
            middle = len(members) // 2
            results = self._post_edit_passage(members[:middle], consecutive) + self._post_edit_passage(
                members[middle:], consecutive
            )
        return _with_shared_usage(results, cost, input_token_count, output_token_count)

    def _build_prompt(self, segment: PostEditSegment) -> str:
        most_similar_first = self._example_retriever.rank(segment.source, self._num_examples)
        least_similar_first: List[Example] = list(reversed(most_similar_first))
        return (
            f"{_format_examples(least_similar_first, self._source_language, self._target_language)}\n\n"
            f"{_format_passage([(0, segment)], self._source_language, self._target_language)}\n\n"
            f"{_format_draft(segment.draft, self._target_language)}\n\n"
            f'Post-edit the draft above for the verse marked role="to_translate".\n'
            f"Source ({self._source_language}):\n"
            f"{escape(segment.source.strip())}"
        )

    def _build_passage_prompt(self, members: Sequence[Tuple[int, PostEditSegment]], consecutive: bool) -> str:
        sources = [segment.source.strip() for _, segment in members]
        most_similar_first = self._example_retriever.rank("\n".join(sources), self._num_examples)
        least_similar_first: List[Example] = list(reversed(most_similar_first))
        if consecutive:
            passages = _format_passage(members, self._source_language, self._target_language, first_number=1)
        else:
            passages = "\n\n".join(
                _format_passage([member], self._source_language, self._target_language, first_number=number)
                for number, member in enumerate(members, 1)
            )
        numbered_sources = "\n".join(f"{number}. {escape(source)}" for number, source in enumerate(sources, 1))
        return (
            f"{_format_examples(least_similar_first, self._source_language, self._target_language)}\n\n"
            f"{passages}\n\n"
            f"{_format_numbered_drafts([segment.draft for _, segment in members], self._target_language)}\n\n"
            f'Post-edit the drafts above for the verses marked role="to_translate".\n'
            f"Source ({self._source_language}):\n"
            f"{numbered_sources}"
        )

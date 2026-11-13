import json
import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from .completion_finish_reason import CompletionFinishReason
from .example import Example
from .example_retriever import ExampleRetriever
from .llm_completion_client import ContentBlockError, LlmCompletionClient, RefusalError
from .post_edit_result import PostEditResult
from .post_edit_segment import PostEditSegment

logger = logging.getLogger(__name__)

DEFAULT_NUM_EXAMPLES = 50
DEFAULT_MAX_ATTEMPTS = 3

_SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert translator and post-editor from {source_language} to {target_language}.

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
errors and omissions and providing your justification for the fix. Read the translation
memory, passage context, and source text; use them as context for post-editing. The
draft is a dynamic translation that has learned the translator's style, so it's
intentionally not a literal translation; don't post-edit it to be one, and go with the
translator's style over the source's style. Favor making more fixes rather than less,
but without mistakes.

Translate the verse marked role="to_translate" and nothing else.

NEVER write the "translation" in any language other than {target_language}.

OUTPUT — one JSON object:
- "changed": true if your text differs from the draft. false otherwise.
- "justification": if your text differs from the draft, one sentence in English — what
  you fixed and why. Otherwise an empty string.
- "translation": your {target_language} text for the verse marked role="to_translate"."""

_RETRY_REMINDER = (
    "\n\nYour previous answer could not be parsed. Reply with a single JSON object and nothing else "
    '(no markdown fence), with exactly the keys "changed", "justification", "translation".'
)

_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "justification": {"type": "string"},
        "translation": {"type": "string"},
    },
    "required": ["changed", "justification", "translation"],
}


def _to_single_line(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())
    return " ".join(line for line in lines if line != "")


def _format_examples(examples: Sequence[Example], source_language: str, target_language: str) -> str:
    lines = ["<translation_memory>"]
    for example in examples:
        lines.append(
            f"  <example>\n"
            f'    <source lang="{source_language}">{example.source.strip()}</source>\n'
            f'    <target lang="{target_language}">{example.target.strip()}</target>\n'
            f"  </example>"
        )
    lines.append("</translation_memory>")
    return "\n".join(lines)


def _format_passage(segment: PostEditSegment, source_language: str, target_language: str) -> str:
    context_by_offset = {verse.offset_from_segment: verse for verse in segment.context_verses if verse.source.strip()}
    offsets = sorted(context_by_offset) + [0]
    lines = ["<passage_context>"]
    for offset in range(min(offsets), max(offsets) + 1):
        if offset == 0:
            lines.append(
                f'  <verse offset="0" role="to_translate">\n'
                f'    <source lang="{source_language}">{segment.source.strip()}</source>\n'
                f"  </verse>"
            )
        elif offset in context_by_offset:
            verse = context_by_offset[offset]
            target = verse.target.strip()
            rendered_target = (
                f'<target lang="{target_language}">{target}</target>'
                if target
                else f'<target lang="{target_language}" status="not_available"/>'
            )
            lines.append(
                f'  <verse offset="{offset}">\n'
                f'    <source lang="{source_language}">{verse.source.strip()}</source>\n'
                f"    {rendered_target}\n"
                f"  </verse>"
            )
        else:
            lines.append(f'  <missing_verse offset="{offset}"/>')
    lines.append("</passage_context>")
    return "\n".join(lines)


def _format_draft(draft: str, target_language: str) -> str:
    draft = draft.strip()
    if not draft:
        return f'<draft_translation lang="{target_language}" status="not_available"/>'
    return f'<draft_translation lang="{target_language}">\n  {draft}\n</draft_translation>'


def _try_parse_response(text: str, draft: str) -> Optional[PostEditResult]:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    translation = _to_single_line(str(data.get("translation", "")))
    if not translation:
        return None
    changed = translation != draft
    return PostEditResult(
        translation=translation,
        succeeded=True,
        changed=changed,
        justification=str(data.get("justification", "")).strip() if changed else "",
    )


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
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self._completion_client = completion_client
        self._example_retriever = example_retriever
        self._source_language = source_language
        self._target_language = target_language
        self._num_examples = num_examples
        self._max_attempts = max_attempts
        self._system_instruction = _SYSTEM_INSTRUCTION_TEMPLATE.format(
            source_language=source_language, target_language=target_language
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
                {"role": "user", "content": prompt if attempt == 1 else prompt + _RETRY_REMINDER},
            ]
            try:
                completion = self._completion_client.complete(messages, response_schema=_RESPONSE_SCHEMA)
            except ContentBlockError:
                logger.warning("Post-edit attempt %d/%d was blocked by content filtering.", attempt, self._max_attempts)
                finish_reason = CompletionFinishReason.CONTENT_FILTER
                break
            except RefusalError:
                logger.warning("Post-edit attempt %d/%d was refused by the model.", attempt, self._max_attempts)
                finish_reason = CompletionFinishReason.REFUSAL
                break
            except ValueError:
                logger.warning(
                    "Post-edit attempt %d/%d got a malformed response.", attempt, self._max_attempts, exc_info=True
                )
                continue
            total_cost += completion.cost
            total_input_token_count += completion.input_token_count
            total_output_token_count += completion.output_token_count
            finish_reason = completion.finish_reason
            result = _try_parse_response(completion.text, draft)
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

    def post_edit_batch(self, segments: Sequence[PostEditSegment]) -> Sequence[PostEditResult]:
        return [self.post_edit(segment) for segment in segments]

    def _build_prompt(self, segment: PostEditSegment) -> str:
        most_similar_first = self._example_retriever.rank(segment.source, self._num_examples)
        least_similar_first: List[Example] = list(reversed(most_similar_first))
        return (
            f"{_format_examples(least_similar_first, self._source_language, self._target_language)}\n\n"
            f"{_format_passage(segment, self._source_language, self._target_language)}\n\n"
            f"{_format_draft(segment.draft, self._target_language)}\n\n"
            f'Post-edit the draft above for the verse marked role="to_translate".\n'
            f"Source ({self._source_language}):\n"
            f"{segment.source.strip()}"
        )

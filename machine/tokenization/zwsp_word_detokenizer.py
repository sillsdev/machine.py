from typing import Any, List

from ..utils.string_utils import is_punctuation
from .latin_word_detokenizer import LatinWordDetokenizer
from .string_detokenizer import DetokenizeOperation


class ZwspWordDetokenizer(LatinWordDetokenizer):
    def _get_operation(self, ctxt: Any, token: str) -> DetokenizeOperation:
        if token[0].isspace():
            return DetokenizeOperation.MERGE_BOTH
        return super()._get_operation(ctxt, token)

    def _get_separator(self, tokens: List[str], ops: List[DetokenizeOperation], index: int) -> str:
        if (
            index < len(tokens) - 1
            and ops[index + 1] == DetokenizeOperation.MERGE_RIGHT
            and is_punctuation(tokens[index + 1][0])
        ):
            return " "
        elif ops[index] == DetokenizeOperation.MERGE_LEFT and is_punctuation(tokens[index][0]):
            return " "
        return "\u200b"

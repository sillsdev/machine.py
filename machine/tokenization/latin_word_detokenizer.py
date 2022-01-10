from collections import deque
from enum import Enum, auto
from typing import Any

from ..utils.string_utils import is_currency_symbol, is_punctuation
from .string_detokenizer import DetokenizeOperation, StringDetokenizer


class QuoteType(Enum):
    DOUBLE_QUOTATION = auto()
    SINGLE_QUOTATION = auto()
    DOUBLE_ANGLE = auto()
    SINGLE_ANGLE = auto()


QUOTATION_MARKS = {
    '"': QuoteType.DOUBLE_QUOTATION,
    "“": QuoteType.DOUBLE_QUOTATION,
    "”": QuoteType.DOUBLE_QUOTATION,
    "„": QuoteType.DOUBLE_QUOTATION,
    "‟": QuoteType.DOUBLE_QUOTATION,
    "'": QuoteType.SINGLE_QUOTATION,
    "‘": QuoteType.SINGLE_QUOTATION,
    "’": QuoteType.SINGLE_QUOTATION,
    "‚": QuoteType.SINGLE_QUOTATION,
    "‛": QuoteType.SINGLE_QUOTATION,
    "«": QuoteType.DOUBLE_ANGLE,
    "»": QuoteType.DOUBLE_ANGLE,
    "‹": QuoteType.SINGLE_ANGLE,
    "›": QuoteType.SINGLE_ANGLE,
}


class LatinWordDetokenizer(StringDetokenizer):
    def _create_context(self) -> Any:
        return deque()

    def _get_operation(self, ctxt: Any, token: str) -> DetokenizeOperation:
        quotes: deque[str] = ctxt
        c = token[0]
        if is_currency_symbol(c) or c in {"(", "[", "{", "¿", "¡", "<"}:
            return DetokenizeOperation.MERGE_RIGHT
        elif c in QUOTATION_MARKS:
            if len(quotes) == 0 or QUOTATION_MARKS[c] != QUOTATION_MARKS[quotes[-1]]:
                quotes.append(c)
                return DetokenizeOperation.MERGE_RIGHT
            else:
                quotes.pop()
                return DetokenizeOperation.MERGE_LEFT
        elif c in {"/", "\\"}:
            return DetokenizeOperation.MERGE_BOTH
        elif is_punctuation(c) or c == ">":
            return DetokenizeOperation.MERGE_LEFT

        return DetokenizeOperation.NO_OPERATION

from enum import Enum, auto
from typing import Union

from . import (
    Detokenizer,
    LatinSentenceTokenizer,
    LatinWordDetokenizer,
    LatinWordTokenizer,
    LineSegmentTokenizer,
    NullTokenizer,
    Tokenizer,
    WhitespaceDetokenizer,
    WhitespaceTokenizer,
    ZwspWordDetokenizer,
    ZwspWordTokenizer,
)


class TokenizerType(Enum):
    NULL = auto()
    LINE_SEGMENT = auto()
    WHITESPACE = auto()
    LATIN = auto()
    LATIN_SENTENCE = auto()
    ZWSP = auto()


def create_tokenizer(type: Union[str, TokenizerType]) -> Tokenizer[str, int, str]:
    if isinstance(type, str):
        type = TokenizerType[type.upper()]
    if type == TokenizerType.NULL:
        return NullTokenizer()
    if type == TokenizerType.LINE_SEGMENT:
        return LineSegmentTokenizer()
    if type == TokenizerType.WHITESPACE:
        return WhitespaceTokenizer()
    if type == TokenizerType.LATIN:
        return LatinWordTokenizer()
    if type == TokenizerType.LATIN_SENTENCE:
        return LatinSentenceTokenizer()
    if type == TokenizerType.ZWSP:
        return ZwspWordTokenizer()


def create_detokenizer(type: Union[str, TokenizerType]) -> Detokenizer[str, str]:
    if isinstance(type, str):
        type = TokenizerType[type.upper()]
    if type == TokenizerType.WHITESPACE:
        return WhitespaceDetokenizer()
    if type == TokenizerType.LATIN:
        return LatinWordDetokenizer()
    if type == TokenizerType.ZWSP:
        return ZwspWordDetokenizer()
    raise RuntimeError(f"Unknown tokenizer: {type}.  Available tokenizers are: whitespace, latin, zwsp.")

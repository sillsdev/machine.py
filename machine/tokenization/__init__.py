from typing import Tuple

from .detokenizer import Detokenizer
from .latin_sentence_tokenizer import LatinSentenceTokenizer
from .latin_word_detokenizer import LatinWordDetokenizer
from .latin_word_tokenizer import LatinWordTokenizer
from .line_segment_tokenizer import LineSegmentTokenizer
from .null_tokenizer import NullTokenizer
from .range_tokenizer import RangeTokenizer
from .string_detokenizer import StringDetokenizer
from .string_tokenizer import StringTokenizer
from .tokenization_utils import get_ranges, split
from .tokenizer import Tokenizer
from .whitespace_detokenizer import WHITESPACE_DETOKENIZER, WhitespaceDetokenizer
from .whitespace_tokenizer import WHITESPACE_TOKENIZER, WhitespaceTokenizer
from .zwsp_word_detokenizer import ZwspWordDetokenizer
from .zwsp_word_tokenizer import ZwspWordTokenizer

__all__ = [
    "Detokenizer",
    "get_ranges",
    "LatinSentenceTokenizer",
    "LatinWordDetokenizer",
    "LatinWordTokenizer",
    "LineSegmentTokenizer",
    "NullTokenizer",
    "RangeTokenizer",
    "split",
    "StringDetokenizer",
    "StringTokenizer",
    "Tokenizer",
    "WHITESPACE_DETOKENIZER",
    "WHITESPACE_TOKENIZER",
    "WhitespaceDetokenizer",
    "WhitespaceTokenizer",
    "ZwspWordDetokenizer",
    "ZwspWordTokenizer",
]

TOKENIZERS = ["latin", "whitespace", "zwsp"]


def get_tokenizer_detokenizer(name: str = "") -> Tuple[Tokenizer, Detokenizer]:
    name_lower = name.lower()
    if "latin" in name_lower or name == "":
        return LatinWordTokenizer(), LatinWordDetokenizer()
    if "whitespace" in name_lower:
        return WhitespaceTokenizer(), WhitespaceDetokenizer()
    if "zwsp" in name_lower:
        return ZwspWordTokenizer(), ZwspWordDetokenizer()
    raise ValueError(
        f"Unknown tokenizer/detokenizer name: {name}.  Available tokenizers are: {TOKENIZERS}.  Defualt tokenizer is latin."
    )

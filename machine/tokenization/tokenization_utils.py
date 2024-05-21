from typing import Generator, Iterable, List

from ..annotations.range import Range
from .detokenizer import Detokenizer
from .latin_word_detokenizer import LatinWordDetokenizer
from .latin_word_tokenizer import LatinWordTokenizer
from .tokenizer import Tokenizer
from .whitespace_detokenizer import WhitespaceDetokenizer
from .whitespace_tokenizer import WhitespaceTokenizer
from .zwsp_word_detokenizer import ZwspWordDetokenizer
from .zwsp_word_tokenizer import ZwspWordTokenizer


def split(s: str, ranges: Iterable[Range[int]]) -> List[str]:
    return [s[range.start : range.end] for range in ranges]


def get_ranges(s: str, tokens: Iterable[str]) -> Generator[Range[int], None, None]:
    start = 0
    for token in tokens:
        index = s.find(token, start)
        if index == -1:
            raise ValueError(f"The string does not contain the specified token: {token}.")
        yield Range.create(index, index + len(token))
        start = index + len(token)


TOKENIZERS = ["latin", "whitespace", "zwsp"]


def create_tokenizer(name: str = "") -> Tokenizer:
    name_lower = name.lower()
    if "latin" == name_lower or name == "":
        return LatinWordTokenizer()
    if "whitespace" == name_lower:
        return WhitespaceTokenizer()
    if "zwsp" == name_lower:
        return ZwspWordTokenizer()
    raise ValueError(
        f"Unknown tokenizer name: {name}.  Available tokenizers are: {TOKENIZERS}.  Defualt tokenizer is latin."
    )


def create_detokenizer(name: str = "") -> Detokenizer:
    name_lower = name.lower()
    if "latin" == name_lower or name == "":
        return LatinWordDetokenizer()
    if "whitespace" == name_lower:
        return WhitespaceDetokenizer()
    if "zwsp" == name_lower:
        return ZwspWordDetokenizer()
    raise ValueError(
        f"Unknown detokenizer name: {name}.  Available detokenizers are: {TOKENIZERS}.  Default detokenizer is latin."
    )

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import regex as re

from ..annotations.range import Range
from ..utils.string_utils import is_control, is_punctuation, is_symbol
from .whitespace_tokenizer import WhitespaceTokenizer

INNER_WORD_PUNCT_REGEX = re.compile(
    r"[&\-.:=,?@\xAD\xB7\u2010\u2011\u2019\u2027]|['_]+",
)
URL_REGEX = re.compile(r"(?:[\w-]+://?|www[.])[^\s()<>]+(?:[\w\d]+|(?:[^\p{P}\s]|/))", re.IGNORECASE)


class LatinWordTokenizer(WhitespaceTokenizer):
    def __init__(self, abbreviations: Iterable[str] = [], treat_apostrophe_as_single_quote: bool = False) -> None:
        self._abbreviations = {a.lower() for a in abbreviations}
        self.treat_apostrophe_as_single_quote = treat_apostrophe_as_single_quote

    def tokenize_as_ranges(self, data: str, data_range: Optional[Range[int]] = None) -> Iterable[Range[int]]:
        if data_range is None:
            data_range = Range.create(0, len(data))
        ctxt = LatinWordTokenizer._TokenizeContext()
        for char_range in super().tokenize_as_ranges(data, data_range):
            url_match = URL_REGEX.match(data[char_range.start : char_range.end])
            if url_match is not None:
                url_len = len(url_match.group())
                yield Range.create(char_range.start, char_range.start + url_len)
                ctxt.index = char_range.start + url_len
            else:
                ctxt.index = char_range.start

            ctxt.word_start = -1
            ctxt.inner_word_punct = -1

            while ctxt.index < char_range.end:
                token_range1, token_range2 = self._process_character(data, data_range, ctxt)
                if token_range1 is not None:
                    yield token_range1
                if token_range2 is not None:
                    yield token_range2

            if ctxt.word_start != -1:
                if ctxt.inner_word_punct != -1:
                    inner_punct_str = data[ctxt.inner_word_punct : char_range.end]
                    if (
                        inner_punct_str == "." and self._is_abbreviation(data, ctxt.word_start, ctxt.inner_word_punct)
                    ) or (inner_punct_str == "'" and not self.treat_apostrophe_as_single_quote):
                        yield Range.create(ctxt.word_start, char_range.end)
                    else:
                        yield Range.create(ctxt.word_start, ctxt.inner_word_punct)
                        yield Range.create(ctxt.inner_word_punct, char_range.end)
                else:
                    yield Range.create(ctxt.word_start, char_range.end)

    def _process_character(
        self, data: str, data_range: Range[int], ctxt: "LatinWordTokenizer._TokenizeContext"
    ) -> Tuple[Optional[Range[int]], Optional[Range[int]]]:
        token_ranges: Tuple[Optional[Range[int]], Optional[Range[int]]] = (None, None)
        c = data[ctxt.index]
        end_index = ctxt.index + 1

        if is_punctuation(c) or is_symbol(c) or is_control(c):
            while end_index != data_range.end and data[end_index] == c:
                end_index += 1
            if ctxt.word_start == -1:
                if c == "'" and not self.treat_apostrophe_as_single_quote:
                    ctxt.word_start = ctxt.index
                else:
                    token_ranges = (Range.create(ctxt.index, end_index), None)
            elif ctxt.inner_word_punct != -1:
                inner_punct_str = data[ctxt.inner_word_punct : ctxt.index]
                if inner_punct_str == "'" and not self.treat_apostrophe_as_single_quote:
                    token_ranges = (Range.create(ctxt.word_start, ctxt.index), None)
                else:
                    token_ranges = (
                        Range.create(ctxt.word_start, ctxt.inner_word_punct),
                        Range.create(ctxt.inner_word_punct, ctxt.index),
                    )
                ctxt.word_start = ctxt.index
            else:
                match = INNER_WORD_PUNCT_REGEX.match(data, ctxt.index)
                if match is not None:
                    ctxt.inner_word_punct = ctxt.index
                    ctxt.index += len(match.group())
                    return token_ranges

                token_ranges = (Range.create(ctxt.word_start, ctxt.index), Range.create(ctxt.index, end_index))
                ctxt.word_start = -1
        elif ctxt.word_start == -1:
            ctxt.word_start = ctxt.index

        ctxt.inner_word_punct = -1
        ctxt.index = end_index
        return token_ranges

    def _is_abbreviation(self, data: str, start: int, end: int) -> bool:
        substr = data[start:end].lower()
        return substr in self._abbreviations

    @dataclass
    class _TokenizeContext:
        index: int = 0
        word_start: int = 0
        inner_word_punct: int = 0

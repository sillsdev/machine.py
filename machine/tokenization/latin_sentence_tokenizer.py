from typing import Iterable, Optional

from ..annotations.range import Range
from ..utils.string_utils import is_delayed_sentence_end, is_sentence_terminal
from .latin_word_tokenizer import LatinWordTokenizer
from .line_segment_tokenizer import LineSegmentTokenizer

LINE_TOKENIZER = LineSegmentTokenizer()


class LatinSentenceTokenizer(LatinWordTokenizer):
    def tokenize_as_ranges(self, data: str, data_range: Optional[Range[int]]) -> Iterable[Range[int]]:
        for line_range in LINE_TOKENIZER.tokenize_as_ranges(data, data_range):
            for sentence_range in self._tokenize_line(data, line_range):
                yield sentence_range

    def _tokenize_line(self, data: str, line_range: Range[int]) -> Iterable[Range[int]]:
        sentence_start = -1
        sentence_end = -1
        in_end = False
        has_end_quote_brackets = False

        for word_range in super().tokenize_as_ranges(data, line_range):
            if sentence_start == -1:
                sentence_start = word_range.start
            word = data[word_range.start : word_range.end]
            if not in_end:
                if is_sentence_terminal(word):
                    in_end = True
            else:
                if is_delayed_sentence_end(word):
                    has_end_quote_brackets = True
                elif has_end_quote_brackets and word[0].islower():
                    in_end = False
                    has_end_quote_brackets = False
                else:
                    yield Range.create(sentence_start, sentence_end)
                    sentence_start = word_range.start
                    in_end = False
                    has_end_quote_brackets = False
            sentence_end = word_range.end

        if sentence_start != -1 and sentence_end != -1:
            yield Range.create(sentence_start, sentence_end if in_end else line_range.end)

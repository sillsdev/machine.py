from typing import Optional, Tuple

from ..annotations.range import Range
from ..utils.string_utils import is_punctuation
from .latin_word_tokenizer import LatinWordTokenizer


class ZwspWordTokenizer(LatinWordTokenizer):
    def _process_character(
        self, data: str, data_range: Range[int], ctxt: LatinWordTokenizer._TokenizeContext
    ) -> Tuple[Optional[Range[int]], Optional[Range[int]]]:
        if data[ctxt.index].isspace():
            end_index = ctxt.index + 1
            while end_index != data_range.end and data[end_index].isspace():
                end_index += 1
            token_ranges: Tuple[Optional[Range[int]], Optional[Range[int]]] = (None, None)
            # ignore whitespace that is followed by whitespace or punctuation
            if ctxt.index != data_range.end - 1 and (is_punctuation(data[end_index]) or data[end_index].isspace()):
                if ctxt.word_start != -1:
                    token_ranges = (Range.create(ctxt.word_start, ctxt.index), None)
                    ctxt.word_start = -1
            # ignore whitespace that is preceded by whitespace or punctuation
            elif ctxt.index != data_range.start and (
                is_punctuation(data[ctxt.index - 1]) or data[ctxt.index - 1].isspace()
            ):
                if ctxt.inner_word_punct != -1:
                    token_ranges = (
                        Range.create(ctxt.word_start, ctxt.inner_word_punct),
                        Range.create(ctxt.inner_word_punct),
                    )
                    ctxt.word_start = -1
            elif ctxt.word_start == -1:
                token_ranges = (Range.create(ctxt.index, end_index), None)
            elif ctxt.inner_word_punct != -1:
                token_ranges = (
                    Range.create(ctxt.word_start, ctxt.inner_word_punct),
                    Range.create(ctxt.inner_word_punct, ctxt.index),
                )
                ctxt.word_start = ctxt.index
            else:
                token_ranges = (Range.create(ctxt.word_start, ctxt.index), Range.create(ctxt.index, end_index))
                ctxt.word_start = -1
            ctxt.inner_word_punct = -1
            ctxt.index = end_index
            return token_ranges
        return super()._process_character(data, data_range, ctxt)

    def _is_whitespace(self, c: str) -> bool:
        return c == "\u200b" or c == "\ufeff"

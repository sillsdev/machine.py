from typing import Iterable, Optional

from ..annotations.range import Range
from .string_tokenizer import StringTokenizer


class WhitespaceTokenizer(StringTokenizer):
    def tokenize_as_ranges(self, data: str, data_range: Optional[Range[int]] = None) -> Iterable[Range[int]]:
        if data_range is None:
            data_range = Range.create(0, len(data))
        start_index = -1
        for i in data_range:
            if self._is_whitespace(data[i]):
                if start_index != -1:
                    yield Range.create(start_index, i)
                start_index = -1
            elif start_index == -1:
                start_index = i

        if start_index != -1:
            yield Range.create(start_index, data_range.end)

    def _is_whitespace(self, c: str) -> bool:
        return c.isspace() or c == "\u200b" or c == "\ufeff"

from typing import Iterable, Optional

from ..annotations.range import Range
from .string_tokenizer import StringTokenizer


class LineSegmentTokenizer(StringTokenizer):
    def tokenize_as_ranges(self, data: str, data_range: Optional[Range[int]] = None) -> Iterable[Range[int]]:
        if data_range is None:
            data_range = Range.create(0, len(data))
        line_start = data_range.start
        i = data_range.start
        while i < data_range.end:
            if data[i] == "\n" or data[i] == "\r":
                yield Range.create(line_start, i)
                if data[i] == "\r" and i + 1 < data_range.end and data[i + 1] == "\n":
                    i += 1
                line_start = i + 1
            i += 1

        if line_start < data_range.end:
            yield Range.create(line_start, data_range.end)

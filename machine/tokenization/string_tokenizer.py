from typing import Iterable, Optional

from ..annotations.range import Range
from .range_tokenizer import RangeTokenizer


class StringTokenizer(RangeTokenizer[str, int, str]):
    def tokenize(self, data: str, data_range: Optional[Range[int]] = None) -> Iterable[str]:
        return (data[r.start : r.end] for r in self.tokenize_as_ranges(data, data_range))

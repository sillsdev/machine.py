from typing import Iterable, Optional

from ..annotations.range import Range, create_range
from .string_tokenizer import StringTokenizer


class NullTokenizer(StringTokenizer):
    def tokenize_as_ranges(self, data: str, data_range: Optional[Range[int]] = None) -> Iterable[Range[int]]:
        if data_range is None:
            data_range = create_range(0, len(data))
        if len(data_range) > 0:
            yield data_range

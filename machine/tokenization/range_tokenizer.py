from abc import abstractmethod
from typing import Iterable, Optional, TypeVar

from ..annotations.range import Range
from .tokenizer import Tokenizer

Data = TypeVar("Data")
Offset = TypeVar("Offset")
Token = TypeVar("Token")


class RangeTokenizer(Tokenizer[Data, Offset, Token]):
    @abstractmethod
    def tokenize_as_ranges(self, data: Data, data_range: Optional[Range[Offset]] = None) -> Iterable[Range[Offset]]:
        ...

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Optional, TypeVar

from ..annotations.range import Range

Data = TypeVar("Data")
Offset = TypeVar("Offset")
Token = TypeVar("Token")


class Tokenizer(ABC, Generic[Data, Offset, Token]):
    @abstractmethod
    def tokenize(self, data: Data, data_range: Optional[Range[Offset]] = None) -> Iterable[Token]:
        ...

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

Data = TypeVar("Data")
Token = TypeVar("Token")


class Detokenizer(ABC, Generic[Data, Token]):
    @abstractmethod
    def detokenize(self, tokens: Iterable[Token]) -> Data:
        ...

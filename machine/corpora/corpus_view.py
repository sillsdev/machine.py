from abc import ABC, abstractmethod
from typing import Generator, Generic, Iterable, TypeVar

from ..utils.context_managed_generator import ContextManagedGenerator

Row = TypeVar("Row")


class CorpusView(ABC, Generic[Row], Iterable[Row]):
    def get_rows(self) -> ContextManagedGenerator[Row, None, None]:
        return ContextManagedGenerator(self._get_rows())

    @abstractmethod
    def _get_rows(self) -> Generator[Row, None, None]:
        ...

    def __iter__(self) -> ContextManagedGenerator[Row, None, None]:
        return self.get_rows()

    def count(self) -> int:
        with self.get_rows() as rows:
            return sum(1 for _ in rows)

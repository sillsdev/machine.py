from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Callable, Generator, Generic, Iterable, Optional, Tuple, TypeVar

from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_utils import get_split_indices

Row = TypeVar("Row")
Item = TypeVar("Item")


class Corpus(ABC, Generic[Row], Iterable[Row]):
    def get_rows(self) -> ContextManagedGenerator[Row, None, None]:
        return ContextManagedGenerator(self._get_rows())

    @abstractmethod
    def _get_rows(self) -> Generator[Row, None, None]:
        ...

    def __iter__(self) -> Generator[Row, None, None]:
        return self._get_rows()

    def count(self) -> int:
        return sum(1 for _ in self)

    def filter(self, predicate: Callable[[Row], bool]) -> "Corpus[Row]":
        return _FilterCorpus(self, lambda row, _: predicate(row))

    def filter_by_index(self, predicate: Callable[[Row, int], bool]) -> "Corpus[Row]":
        return _FilterCorpus(self, predicate)

    def take(self, count: int) -> "Corpus[Row]":
        return _TakeCorpus(self, count)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, seed: Any = None
    ) -> Tuple["Corpus[Row]", "Corpus[Row]", int, int]:
        corpus_size = self.count()
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        main_corpus = self.filter_by_index(lambda _, i: i not in split_indices)
        split_corpus = self.filter_by_index(lambda _, i: i in split_indices)

        return main_corpus, split_corpus, corpus_size - len(split_indices), len(split_indices)

    def map(self, selector: Callable[[Row], Item]) -> ContextManagedGenerator[Item, None, None]:
        return ContextManagedGenerator(self._map(selector))

    def _map(self, selector: Callable[[Row], Item]) -> Generator[Item, None, None]:
        yield from (selector(row) for row in self)


class _FilterCorpus(Corpus[Row]):
    def __init__(self, corpus: Corpus[Row], predicate: Callable[[Row, int], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[Row, None, None]:
        yield from (row for i, row in enumerate(self._corpus) if self._predicate(row, i))


class _TakeCorpus(Corpus[Row]):
    def __init__(self, corpus: Corpus, count: int) -> None:
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[Row, None, None]:
        yield from islice(self._corpus, self._count)

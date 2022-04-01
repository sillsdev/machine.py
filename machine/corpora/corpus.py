from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Callable, Generator, Generic, Iterable, Optional, Tuple, TypeVar

from ..utils.context_managed_generator import ContextManagedGenerator
from .alignment_row import AlignmentRow
from .corpora_utils import get_split_indices
from .parallel_text_row import ParallelTextRow
from .text_row import TextRow

Row = TypeVar("Row", TextRow, ParallelTextRow, AlignmentRow)
Item = TypeVar("Item")


class Corpus(ABC, Generic[Row], Iterable[Row]):
    def get_rows(self) -> ContextManagedGenerator[Row, None, None]:
        return ContextManagedGenerator(self._get_rows())

    @abstractmethod
    def _get_rows(self) -> Generator[Row, None, None]:
        ...

    def __iter__(self) -> ContextManagedGenerator[Row, None, None]:
        return self.get_rows()

    @property
    def missing_rows_allowed(self) -> bool:
        return True

    def count(self, include_empty: bool = True) -> int:
        with self.get_rows() as rows:
            return sum(1 for row in rows if include_empty or not row.is_empty)

    def filter_empty(self) -> "Corpus[Row]":
        return self.filter(lambda r: not r.is_empty)

    def filter(self, predicate: Callable[[Row], bool]) -> "Corpus[Row]":
        return self.filter_by_index(lambda r, _: predicate(r))

    def filter_by_index(self, predicate: Callable[[Row, int], bool]) -> "Corpus[Row]":
        return _FilterCorpus(self, predicate)

    def take(self, count: int) -> "Corpus[Row]":
        return _TakeCorpus(self, count)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, include_empty: bool = True, seed: Any = None
    ) -> Tuple["Corpus[Row]", "Corpus[Row]", int, int]:
        corpus_size = self.count(include_empty)
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        main_corpus = self.filter_by_index(lambda r, i: i not in split_indices and (include_empty or not r.is_empty))
        split_corpus = self.filter_by_index(lambda r, i: i in split_indices and (include_empty or not r.is_empty))

        return main_corpus, split_corpus, corpus_size - len(split_indices), len(split_indices)

    def interleaved_split(
        self, percent: Optional[float] = None, size: Optional[int] = None, include_empty: bool = True, seed: Any = None
    ) -> Tuple[ContextManagedGenerator[Tuple[Row, bool], None, None], int, int]:
        corpus_size = self.count(include_empty)
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        corpus = self
        if not include_empty:
            corpus = self.filter(lambda r: not r.is_empty)
        return (
            corpus.map_by_index(lambda r, i: (r, i in split_indices)),
            corpus_size - len(split_indices),
            len(split_indices),
        )

    def map(self, selector: Callable[[Row], Item]) -> ContextManagedGenerator[Item, None, None]:
        return ContextManagedGenerator(self._map_by_index(lambda r, _: selector(r)))

    def map_by_index(self, selector: Callable[[Row, int], Item]) -> ContextManagedGenerator[Item, None, None]:
        return ContextManagedGenerator(self._map_by_index(selector))

    def _map_by_index(self, selector: Callable[[Row, int], Item]) -> Generator[Item, None, None]:
        with self.get_rows() as rows:
            yield from (selector(row, i) for i, row in enumerate(rows))


class _FilterCorpus(Corpus[Row]):
    def __init__(self, corpus: Corpus[Row], predicate: Callable[[Row, int], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[Row, None, None]:
        with self._corpus.get_rows() as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class _TakeCorpus(Corpus[Row]):
    def __init__(self, corpus: Corpus, count: int) -> None:
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[Row, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)

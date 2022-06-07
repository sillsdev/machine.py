from __future__ import annotations

from abc import abstractmethod
from itertools import islice
from typing import Callable, Generator, Iterable, Optional

from ..utils.context_managed_generator import ContextManagedGenerator
from .alignment_collection import AlignmentCollection
from .alignment_row import AlignmentRow
from .corpus import Corpus


class AlignmentCorpus(Corpus[AlignmentRow]):
    @property
    @abstractmethod
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        ...

    def get_rows(self, text_ids: Optional[Iterable[str]] = None) -> ContextManagedGenerator[AlignmentRow, None, None]:
        return ContextManagedGenerator(self._get_rows(text_ids))

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[AlignmentRow, None, None]:
        alignment_collection_id_set = set((t.id for t in self.alignment_collections) if text_ids is None else text_ids)
        for tac in self.alignment_collections:
            if tac.id in alignment_collection_id_set:
                with tac.get_rows() as rows:
                    yield from rows

    @property
    def missing_rows_allowed(self) -> bool:
        return any(ac.missing_rows_allowed for ac in self.alignment_collections)

    def count(self, include_empty: bool = True) -> int:
        return sum(ac.count(include_empty) for ac in self.alignment_collections)

    def invert(self) -> AlignmentCorpus:
        def _invert(row: AlignmentRow) -> AlignmentRow:
            return row.invert()

        return self.transform(_invert)

    def transform(self, transform: Callable[[AlignmentRow], AlignmentRow]) -> AlignmentCorpus:
        return _TransformAlignmentCorpus(self, transform)

    def filter_nonempty(self) -> AlignmentCorpus:
        return self.filter(lambda r: not r.is_empty)

    def filter(self, predicate: Callable[[AlignmentRow], bool]) -> AlignmentCorpus:
        return self.filter_by_index(lambda r, _: predicate(r))

    def filter_by_index(self, predicate: Callable[[AlignmentRow, int], bool]) -> AlignmentCorpus:
        return _FilterAlignmentCorpus(self, predicate)

    def take(self, count: int) -> AlignmentCorpus:
        return _TakeAlignmentCorpus(self, count)


class _TransformAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, transform: Callable[[AlignmentRow], AlignmentRow]):
        self._corpus = corpus
        self._transform = transform

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    @property
    def missing_rows_allowed(self) -> bool:
        return self._corpus.missing_rows_allowed

    def count(self, include_empty: bool = True) -> int:
        return self._corpus.count(include_empty)

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            yield from map(self._transform, rows)


class _FilterAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, predicate: Callable[[AlignmentRow, int], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class _TakeAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, count: int) -> None:
        self._corpus = corpus
        self._count = count

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            yield from islice(rows, self._count)

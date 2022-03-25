from abc import ABC, abstractmethod
from itertools import islice
from typing import Callable, Generator, Iterable, Optional

from ..utils.context_managed_generator import ContextManagedGenerator
from .alignment_collection import AlignmentCollection
from .alignment_row import AlignmentRow


class AlignmentCorpus(ABC, Iterable[AlignmentRow]):
    @property
    @abstractmethod
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        ...

    def get_rows(
        self, alignment_collection_ids: Optional[Iterable[str]] = None
    ) -> ContextManagedGenerator[AlignmentRow, None, None]:
        return ContextManagedGenerator(self._get_rows(alignment_collection_ids))

    def _get_rows(self, alignment_collection_ids: Optional[Iterable[str]]) -> Generator[AlignmentRow, None, None]:
        alignment_collection_id_set = set(
            (t.id for t in self.alignment_collections) if alignment_collection_ids is None else alignment_collection_ids
        )
        for tac in self.alignment_collections:
            if tac.id in alignment_collection_id_set:
                with tac.get_rows() as rows:
                    yield from rows

    def __iter__(self) -> ContextManagedGenerator[AlignmentRow, None, None]:
        return self.get_rows()

    def count(self) -> int:
        with self.get_rows() as rows:
            return sum(1 for _ in rows)

    def invert(self) -> "AlignmentCorpus":
        def _invert(row: AlignmentRow) -> AlignmentRow:
            return row.invert()

        return self.transform(_invert)

    def filter(self, predicate: Callable[[AlignmentRow], bool]) -> "AlignmentCorpus":
        return _FilterAlignmentCorpus(self, predicate)

    def transform(self, transform: Callable[[AlignmentRow], AlignmentRow]) -> "AlignmentCorpus":
        return _TransformAlignmentCorpus(self, transform)

    def take(self, count: int) -> "AlignmentCorpus":
        return _TakeAlignmentCorpus(self, count)


class _TransformAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, transform: Callable[[AlignmentRow], AlignmentRow]):
        self._corpus = corpus
        self._transform = transform

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from map(self._transform, rows)


class _FilterAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, predicate: Callable[[AlignmentRow], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from filter(self._predicate, rows)


class _TakeAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, count: int) -> None:
        self._corpus = corpus
        self._count = count

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)

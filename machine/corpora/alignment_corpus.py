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
    def alignment_collections(self) -> Iterable[AlignmentCollection]: ...

    def get_rows(self, text_ids: Optional[Iterable[str]] = None) -> ContextManagedGenerator[AlignmentRow, None, None]:
        return ContextManagedGenerator(self._get_rows(text_ids))

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[AlignmentRow, None, None]:
        alignment_collection_id_set = set((t.id for t in self.alignment_collections) if text_ids is None else text_ids)
        for tac in self.alignment_collections:
            if tac.id in alignment_collection_id_set:
                with tac.get_rows() as rows:
                    yield from rows

    def count(self, include_empty: bool = True, text_ids: Optional[Iterable[str]] = None) -> int:
        with self.get_rows(text_ids) as rows:
            return sum(1 for row in rows if include_empty or not row.is_empty)

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

    def filter_texts(self, filter: Optional[Iterable[str]]) -> AlignmentCorpus:
        if not filter:
            return self
        return _FilterTextsAlignmentCorpus(self, filter)

    def take(self, count: int) -> AlignmentCorpus:
        return _TakeAlignmentCorpus(self, count)


class _TransformAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, transform: Callable[[AlignmentRow], AlignmentRow]):
        self._corpus = corpus
        self._transform = transform

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    def count(self, include_empty: bool = True, text_ids: Optional[Iterable[str]] = None) -> int:
        return self._corpus.count(include_empty, text_ids)

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


class _FilterTextsAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, text_ids: Iterable[str]) -> None:
        self._corpus = corpus
        self._text_ids = set(text_ids)

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return (ac for ac in self._corpus.alignment_collections if ac.id in self._text_ids)

    def count(self, include_empty: bool = True, text_ids: Optional[Iterable[str]] = None) -> int:
        return self._corpus.count(
            include_empty, self._text_ids if text_ids is None else self._text_ids.intersection(text_ids)
        )

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[AlignmentRow, None, None]:
        yield from self._corpus._get_rows(self._text_ids if text_ids is None else self._text_ids.intersection(text_ids))

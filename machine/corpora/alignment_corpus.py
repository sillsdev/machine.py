from abc import abstractmethod
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

    def get_rows(
        self, alignment_collection_ids: Optional[Iterable[str]] = None
    ) -> ContextManagedGenerator[AlignmentRow, None, None]:
        return ContextManagedGenerator(self._get_rows(alignment_collection_ids))

    def _get_rows(
        self, alignment_collection_ids: Optional[Iterable[str]] = None
    ) -> Generator[AlignmentRow, None, None]:
        alignment_collection_id_set = set(
            (t.id for t in self.alignment_collections) if alignment_collection_ids is None else alignment_collection_ids
        )
        for tac in self.alignment_collections:
            if tac.id in alignment_collection_id_set:
                yield from tac

    def invert(self) -> "AlignmentCorpus":
        def _invert(row: AlignmentRow) -> AlignmentRow:
            return row.invert()

        return self.transform(_invert)

    def transform(self, transform: Callable[[AlignmentRow], AlignmentRow]) -> "AlignmentCorpus":
        return _TransformAlignmentCorpus(self, transform)


class _TransformAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpus: AlignmentCorpus, transform: Callable[[AlignmentRow], AlignmentRow]):
        self._corpus = corpus
        self._transform = transform

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return self._corpus.alignment_collections

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        yield from map(self._transform, self._corpus)

from abc import abstractmethod
from typing import Generator, Iterable, Optional

from .alignment_collection import AlignmentCollection
from .alignment_corpus_view import AlignmentCorpusView
from .alignment_row import AlignmentRow


class AlignmentCorpus(AlignmentCorpusView):
    @property
    @abstractmethod
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        ...

    @abstractmethod
    def __getitem__(self, id: str) -> Optional[AlignmentCollection]:
        ...

    def get_alignment_collection(self, id: str) -> Optional[AlignmentCollection]:
        return self[id]

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        for tac in self.alignment_collections:
            with tac.get_rows() as rows:
                yield from rows

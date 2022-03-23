from abc import abstractmethod
from typing import Iterable, Optional

from .alignment_collection import AlignmentCollection
from .alignment_corpus_view import AlignmentCorpusView


class AlignmentCorpus(AlignmentCorpusView):
    @property
    @abstractmethod
    def text_alignment_collections(self) -> Iterable[AlignmentCollection]:
        ...

    @abstractmethod
    def __getitem__(self, id: str) -> Optional[AlignmentCollection]:
        ...

    def get_text_alignment_collection(self, id: str) -> Optional[AlignmentCollection]:
        return self[id]

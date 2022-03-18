from abc import abstractmethod
from typing import Iterable, Optional

from .text_alignment_collection import TextAlignmentCollection
from .text_alignment_corpus_view import TextAlignmentCorpusView


class TextAlignmentCorpus(TextAlignmentCorpusView):
    @property
    def source(self) -> TextAlignmentCorpusView:
        return self

    @property
    @abstractmethod
    def text_alignment_collections(self) -> Iterable[TextAlignmentCollection]:
        ...

    @abstractmethod
    def __getitem__(self, id: str) -> Optional[TextAlignmentCollection]:
        ...

    def get_text_alignment_collection(self, id: str) -> Optional[TextAlignmentCollection]:
        return self[id]

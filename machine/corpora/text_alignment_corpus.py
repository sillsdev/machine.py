from abc import ABC, abstractmethod
from typing import Iterable

from .text_alignment_collection import TextAlignmentCollection


class TextAlignmentCorpus(ABC):
    @property
    @abstractmethod
    def text_alignment_collections(self) -> Iterable[TextAlignmentCollection]:
        ...

    @abstractmethod
    def __getitem__(self, id: str) -> TextAlignmentCollection:
        ...

    @abstractmethod
    def create_null_text_alignment_collection(self, id: str) -> TextAlignmentCollection:
        ...

    @abstractmethod
    def invert(self) -> "TextAlignmentCorpus":
        ...

    def get_text_alignment_collection(self, id: str) -> TextAlignmentCollection:
        return self[id]

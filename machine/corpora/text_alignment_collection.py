from abc import ABC, abstractmethod
from typing import Iterable

from .text_alignment import TextAlignment


class TextAlignmentCollection(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @property
    @abstractmethod
    def sort_key(self) -> str:
        ...

    @property
    @abstractmethod
    def alignments(self) -> Iterable[TextAlignment]:
        ...

    @abstractmethod
    def invert(self) -> "TextAlignmentCollection":
        ...

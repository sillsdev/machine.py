from abc import ABC, abstractmethod
from typing import Generator

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
    def alignments(self) -> Generator[TextAlignment, None, None]:
        ...

    @abstractmethod
    def invert(self) -> "TextAlignmentCollection":
        ...

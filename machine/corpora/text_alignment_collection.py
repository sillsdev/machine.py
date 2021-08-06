from abc import ABC, abstractmethod

from ..utils.context_managed_generator import ContextManagedGenerator
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
    def alignments(self) -> ContextManagedGenerator[TextAlignment, None, None]:
        ...

    @abstractmethod
    def invert(self) -> "TextAlignmentCollection":
        ...

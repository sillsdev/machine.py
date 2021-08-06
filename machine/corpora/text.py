from abc import ABC, abstractmethod

from ..utils.context_managed_generator import ContextManagedGenerator
from .text_segment import TextSegment


class Text(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @property
    @abstractmethod
    def sort_key(self) -> str:
        ...

    @abstractmethod
    def get_segments(self, include_text: bool = True) -> ContextManagedGenerator[TextSegment, None, None]:
        ...

from abc import ABC, abstractmethod
from typing import Generator, Iterable, Optional

from ..utils.context_managed_generator import ContextManagedGenerator
from .text import Text
from .text_segment import TextSegment


class TextCorpus(ABC):
    @property
    @abstractmethod
    def texts(self) -> Iterable[Text]:
        ...

    @abstractmethod
    def get_text(self, id: str) -> Optional[Text]:
        ...

    @abstractmethod
    def get_text_sort_key(self, id: str) -> str:
        ...

    def get_segments(self, include_text: bool = True) -> ContextManagedGenerator[TextSegment, None, None]:
        return ContextManagedGenerator(self._get_segments(include_text))

    def _get_segments(self, include_text: bool) -> Generator[TextSegment, None, None]:
        for text in self.texts:
            with text.get_segments(include_text) as segments:
                for segment in segments:
                    yield segment

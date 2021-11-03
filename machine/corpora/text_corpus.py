from abc import ABC, abstractmethod
from typing import Generator, Iterable

from ..utils.context_managed_generator import ContextManagedGenerator
from .text import Text
from .text_segment import TextSegment


class TextCorpus(ABC):
    @property
    @abstractmethod
    def texts(self) -> Iterable[Text]:
        ...

    @abstractmethod
    def __getitem__(self, id: str) -> Text:
        ...

    @abstractmethod
    def create_null_text(self, id: str) -> Text:
        ...

    def get_text(self, id: str) -> Text:
        return self[id]

    def get_segments(self, include_text: bool = True) -> ContextManagedGenerator[TextSegment, None, None]:
        return ContextManagedGenerator(self._get_segments(include_text))

    def _get_segments(self, include_text: bool) -> Generator[TextSegment, None, None]:
        for text in self.texts:
            with text.get_segments(include_text) as segments:
                for segment in segments:
                    yield segment

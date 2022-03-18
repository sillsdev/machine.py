from abc import ABC, abstractmethod
from typing import Generator

from ..utils.context_managed_generator import ContextManagedGenerator
from .text_alignment_corpus_row import TextAlignmentCorpusRow


class TextAlignmentCollection(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @property
    @abstractmethod
    def sort_key(self) -> str:
        ...

    def get_rows(self) -> ContextManagedGenerator[TextAlignmentCorpusRow, None, None]:
        return ContextManagedGenerator(self._get_rows())

    @abstractmethod
    def _get_rows(self) -> Generator[TextAlignmentCorpusRow, None, None]:
        ...

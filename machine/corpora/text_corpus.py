from abc import ABC, abstractmethod
from typing import Iterable, Optional
from .text import Text


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

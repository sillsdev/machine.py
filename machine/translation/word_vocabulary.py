from abc import ABC, abstractmethod
from typing import Optional, Sequence


class WordVocabulary(ABC, Sequence[str]):
    @abstractmethod
    def index(self, word: Optional[str]) -> int:
        ...

from abc import ABC, abstractmethod
from typing import Sequence


class WordVocabulary(ABC, Sequence[str]):
    @abstractmethod
    def get_index(self, word: str) -> int:
        ...

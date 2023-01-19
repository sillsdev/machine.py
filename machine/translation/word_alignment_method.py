from abc import abstractmethod
from typing import Callable, Optional, Sequence

from .word_aligner import WordAligner


class WordAlignmentMethod(WordAligner):
    @property
    @abstractmethod
    def score_selector(self) -> Optional[Callable[[Sequence[str], int, Sequence[str], int], float]]:
        ...

    @score_selector.setter
    @abstractmethod
    def score_selector(self, value: Optional[Callable[[Sequence[str], int, Sequence[str], int], float]]) -> None:
        ...

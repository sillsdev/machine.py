from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from .translation_result import TranslationResult


class TranslationEngine(ABC):
    @abstractmethod
    def translate(self, segment: Sequence[str]) -> TranslationResult:
        ...

    @abstractmethod
    def translate_n(self, n: int, segment: Sequence[str]) -> Iterable[TranslationResult]:
        ...

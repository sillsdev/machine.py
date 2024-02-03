from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from .translation_result import TranslationResult
from .translation_suggestion import TranslationSuggestion


class TranslationSuggester(ABC):
    def __init__(self, confidence_threshold: float = 0, break_on_punctuation: bool = True) -> None:
        self.confidence_threshold = confidence_threshold
        self.break_on_punctuation = break_on_punctuation

    @abstractmethod
    def get_suggestions(
        self, n: int, prefix_count: int, is_last_word_complete: bool, results: Iterable[TranslationResult]
    ) -> Sequence[TranslationSuggestion]: ...

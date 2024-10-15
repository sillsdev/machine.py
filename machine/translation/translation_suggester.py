from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

from .interactive_translator import InteractiveTranslator
from .translation_result import TranslationResult
from .translation_suggestion import TranslationSuggestion
from .truecaser import Truecaser


class TranslationSuggester(ABC):
    def __init__(self, confidence_threshold: float = 0, break_on_punctuation: bool = True) -> None:
        self.confidence_threshold = confidence_threshold
        self.break_on_punctuation = break_on_punctuation

    @abstractmethod
    def get_suggestions(
        self, n: int, prefix_count: int, is_last_word_complete: bool, results: Iterable[TranslationResult]
    ) -> Sequence[TranslationSuggestion]: ...

    def get_suggestions_from_translator(
        self, n: int, translator: InteractiveTranslator, truecaser: Optional[Truecaser] = None
    ) -> Sequence[TranslationSuggestion]:
        results = translator.get_current_results()
        if truecaser is not None:
            results = (
                truecaser.truecase_translation_result(result, translator.target_detokenizer) for result in results
            )
        return self.get_suggestions(n, len(translator.prefix_word_ranges), translator.is_last_word_complete, results)

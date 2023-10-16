from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .translation_result import TranslationResult


@dataclass
class TranslationSuggestion:
    result: TranslationResult
    target_word_indices: Sequence[int] = field(default_factory=list)
    confidence: float = 0

    @property
    def target_words(self) -> Iterable[str]:
        return (self.result.target_tokens[i] for i in self.target_word_indices)

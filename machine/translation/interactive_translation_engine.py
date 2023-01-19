from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from .translation_engine import TranslationEngine
from .word_graph import WordGraph


class InterativeTranslationEngine(TranslationEngine):
    @abstractmethod
    def get_word_graph(self, segment: Sequence[str]) -> WordGraph:
        ...

    def train_segment(
        self, source_segment: Sequence[str], target_segment: Sequence[str], sentence_start: bool = True
    ) -> None:
        ...

    def __enter__(self) -> InterativeTranslationEngine:
        return self

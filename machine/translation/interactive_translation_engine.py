from __future__ import annotations

from abc import abstractmethod
from typing import Sequence, Union

from .translation_engine import TranslationEngine
from .word_graph import WordGraph


class InterativeTranslationEngine(TranslationEngine):
    @abstractmethod
    def get_word_graph(self, segment: Union[str, Sequence[str]]) -> WordGraph:
        ...

    def train_segment(
        self,
        source_segment: Union[str, Sequence[str]],
        target_segment: Union[str, Sequence[str]],
        sentence_start: bool = True,
    ) -> None:
        ...

    def __enter__(self) -> InterativeTranslationEngine:
        return self

from abc import ABC, abstractmethod
from typing import Sequence

from .translation_result_builder import TranslationResultBuilder
from .word_graph import WordGraph


class WordConfidenceEstimator(ABC):
    @abstractmethod
    def estimate_word_graph(self, source_segment: Sequence[str], word_graph: WordGraph) -> None:
        ...

    @abstractmethod
    def estimate_translation_result(self, source_segment: Sequence[str], builder: TranslationResultBuilder) -> None:
        ...

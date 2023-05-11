from abc import ABC, abstractmethod

from ..annotations.range import Range


class WordConfidenceEstimator(ABC):
    @abstractmethod
    def estimate(self, source_segment_range: Range[int], target_word: str) -> float:
        ...

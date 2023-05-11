from typing import Callable, Optional, Sequence

from ..annotations.range import Range
from .word_confidence_estimator import WordConfidenceEstimator


class Ibm1WordConfidenceEstimator(WordConfidenceEstimator):
    def __init__(
        self,
        get_translation_prob: Callable[[Optional[str], Optional[str]], float],
        source_tokens: Sequence[str],
        phrase_only: bool = True,
    ) -> None:
        self._get_translation_prob = get_translation_prob
        self._source_tokens = source_tokens
        self.phrase_only = phrase_only

    def estimate(self, source_segment_range: Range[int], target_word: str) -> float:
        if not self.phrase_only:
            source_segment_range = Range[int].create(0, len(self._source_tokens))
        max_confidence = self._get_translation_prob(None, target_word)
        for i in source_segment_range:
            confidence = self._get_translation_prob(self._source_tokens[i], target_word)
            if confidence > max_confidence:
                max_confidence = confidence
        return max_confidence

from typing import Callable, Optional, Sequence

from ..annotations.range import Range
from .translation_result_builder import TranslationResultBuilder
from .word_confidence_estimator import WordConfidenceEstimator
from .word_graph import WordGraph


class Ibm1WordConfidenceEstimator(WordConfidenceEstimator):
    def __init__(
        self, get_translation_prob: Callable[[Optional[str], Optional[str]], float], phrase_only: bool = True
    ) -> None:
        self._get_translation_prob = get_translation_prob
        self.phrase_only = phrase_only

    def estimate_word_graph(self, source_segment: Sequence[str], word_graph: WordGraph) -> None:
        phrase_range = Range[int].create(0, len(source_segment))
        for arc in word_graph.arcs:
            if self.phrase_only:
                phrase_range = arc.source_segment_range

            for k in range(len(arc.words)):
                arc.word_confidences[k] = self._get_confidence(source_segment, phrase_range, arc.words[k])

    def estimate_translation_result(self, source_segment: Sequence[str], builder: TranslationResultBuilder) -> None:
        phrase_range = Range[int].create(0, len(source_segment))
        start_index = 0
        for phrase in builder.phrases:
            if self.phrase_only:
                phrase_range = phrase.source_segment_range

            for j in range(start_index, phrase.target_cut):
                confidence = self._get_confidence(source_segment, phrase_range, builder.words[j])
                builder.set_confidence(j, confidence)

    def _get_confidence(self, source_segment: Sequence[str], phrase_range: Range[int], target_word: str) -> float:
        max_confidence = self._get_translation_prob(None, target_word)
        for i in phrase_range:
            confidence = self._get_translation_prob(source_segment[i], target_word)
            if confidence > max_confidence:
                max_confidence = confidence
        return max_confidence

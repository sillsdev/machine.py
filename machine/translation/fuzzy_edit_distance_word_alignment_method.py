from statistics import mean
from typing import Callable, Iterable, Optional, Sequence, Tuple

from ..sequence_alignment.pairwise_alignment_algorithm import AlignmentMode, PairwiseAlignmentAlgorithm
from ..statistics.log_space import log_space_multiple, to_log_space
from .segment_scorer import SegmentScorer
from .word_alignment_matrix import WordAlignmentMatrix
from .word_alignment_method import WordAlignmentMethod

_DEFAULT_ALPHA = 0.2
_DEFAULT_MAX_DISTANCE = 3


def _get_word_indices(sequence: Sequence[str]) -> Tuple[Iterable[int], int, int]:
    return range(len(sequence)), 0, len(sequence)


def _compute_distance_score(i1: int, i2: int, source_length: int) -> float:
    if source_length == 1:
        return 0.1
    return abs(i1 - i2) / (source_length - 1)


class FuzzyEditDistanceWordAlignmentMethod(WordAlignmentMethod):
    def __init__(self) -> None:
        self._score_selector: Optional[Callable[[Sequence[str], int, Sequence[str], int], float]] = None
        self._scorer: Optional[SegmentScorer] = None
        self.max_distance = _DEFAULT_MAX_DISTANCE
        self.alpha = _DEFAULT_ALPHA

    @property
    def score_selector(self) -> Optional[Callable[[Sequence[str], int, Sequence[str], int], float]]:
        return self._score_selector

    @score_selector.setter
    def score_selector(self, value: Optional[Callable[[Sequence[str], int, Sequence[str], int], float]]) -> None:
        self._score_selector = value
        self._scorer = None if self._score_selector is None else SegmentScorer(self._score_selector)

    def align(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        if self._scorer is None or self._score_selector is None:
            raise RuntimeError("A score selector has not been assigned.")

        paa = PairwiseAlignmentAlgorithm(
            self._scorer,
            source_segment,
            target_segment,
            _get_word_indices,
            mode=AlignmentMode.GLOBAL,
            expansion_compression_enabled=True,
            transposition_enabled=True,
        )
        paa.compute()
        alignment = next(iter(paa.get_alignments()))
        wa_matrix = WordAlignmentMatrix.from_word_pairs(len(source_segment), len(target_segment))
        for c in range(alignment.column_count):
            for j in alignment[1, c]:
                if alignment[0, c].is_null:
                    prob = self._score_selector(source_segment, -1, target_segment, j)
                    best_score = self._compute_alignment_score(prob, 0)
                    tc = c - 1
                    while tc >= 0 and alignment[0, tc].is_null:
                        tc -= 1
                    i = 0 if tc == -1 else alignment[0, tc].last
                    min_index = i
                    max_index = i + 1
                else:
                    prob = mean(self._score_selector(source_segment, i, target_segment, j) for i in alignment[0, c])
                    best_score = self._compute_alignment_score(prob, 0)
                    min_index = alignment[0, c].first - 1
                    max_index = alignment[0, c].last + 1

                best_index = -1
                for i in reversed(range(max(0, min_index - self.max_distance), min_index + 1)):
                    prob = self._score_selector(source_segment, i, target_segment, j)
                    distance_score = _compute_distance_score(i, min_index + 1, len(source_segment))
                    score = self._compute_alignment_score(prob, distance_score)
                    if score > best_score:
                        best_score = score
                        best_index = i

                for i in range(max_index, min(len(source_segment), max_index + self.max_distance)):
                    prob = self._score_selector(source_segment, i, target_segment, j)
                    distance_score = _compute_distance_score(i, max_index - 1, len(source_segment))
                    score = self._compute_alignment_score(prob, distance_score)
                    if score > best_score:
                        best_score = score
                        best_index = i

                if best_index == -1:
                    if not alignment[0, c].is_null:
                        wa_matrix[(min_index + 1), j] = True
                        wa_matrix[(max_index - 1), j] = True
                else:
                    wa_matrix[best_index, j] = True

        return wa_matrix

    def align_batch(self, segments: Sequence[Sequence[Sequence[str]]]) -> Sequence[WordAlignmentMatrix]:
        return [self.align(source_segment, target_segment) for source_segment, target_segment in segments]

    def _compute_alignment_score(self, probability: float, distance_score: float) -> float:
        return log_space_multiple(
            to_log_space(probability) * self.alpha, to_log_space(1.0 - distance_score) * (1.0 - self.alpha)
        )

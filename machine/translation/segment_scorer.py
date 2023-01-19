from typing import Callable, Optional, Sequence

from ..sequence_alignment.pairwise_alignment_scorer import PairwiseAlignmentScorer

MAX_VALUE = 100000


class SegmentScorer(PairwiseAlignmentScorer[Sequence[str], int]):
    def __init__(self, score_selector: Callable[[Sequence[str], int, Sequence[str], int], float]) -> None:
        self._score_selector = score_selector

    def get_gap_penalty(self, sequence1: Sequence[str], sequence2: Sequence[str]) -> int:
        return -(MAX_VALUE // 10)

    def get_insertion_score(self, sequence1: Sequence[str], p: Optional[int], sequence2: Sequence[str], q: int) -> int:
        return int(self._score_selector(sequence1, -1, sequence2, q) * MAX_VALUE)

    def get_deletion_score(self, sequence1: Sequence[str], p: int, sequence2: Sequence[str], q: Optional[int]) -> int:
        return int(self._score_selector(sequence1, p, sequence2, -1) * MAX_VALUE)

    def get_substitution_score(self, sequence1: Sequence[str], p: int, sequence2: Sequence[str], q: int) -> int:
        source_word = sequence1[p]
        target_word = sequence2[q]
        return int(
            (1.0 if source_word == target_word else self._score_selector(sequence1, p, sequence2, q)) * MAX_VALUE
        )

    def get_expansion_score(self, sequence1: Sequence[str], p: int, sequence2: Sequence[str], q1: int, q2: int) -> int:
        return (
            self.get_substitution_score(sequence1, p, sequence2, q1)
            + self.get_substitution_score(sequence1, p, sequence2, q2)
        ) // 2

    def get_compression_score(
        self, sequence1: Sequence[str], p1: int, p2: int, sequence2: Sequence[str], q: int
    ) -> int:
        return (
            self.get_substitution_score(sequence1, p1, sequence2, q)
            + self.get_substitution_score(sequence1, p2, sequence2, q)
        ) // 2

    def get_transposition_score(
        self, sequence1: Sequence[str], p1: int, p2: int, sequence2: Sequence[str], q1: int, q2: int
    ) -> int:
        return (
            self.get_substitution_score(sequence1, p1, sequence2, q2)
            + self.get_substitution_score(sequence1, p2, sequence2, q1)
        ) // 2

    def get_max_score1(self, sequence1: Sequence[str], p: int, sequence2: Sequence[str]) -> int:
        return MAX_VALUE

    def get_max_score2(self, sequence1: Sequence[str], sequence2: Sequence[str], q: int) -> int:
        return MAX_VALUE

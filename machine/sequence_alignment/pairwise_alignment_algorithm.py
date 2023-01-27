import sys
from enum import Enum, auto
from typing import Callable, Generic, Iterable, List, Tuple, TypeVar

from .alignment import Alignment
from .alignment_cell import AlignmentCell
from .pairwise_alignment_scorer import PairwiseAlignmentScorer

Seq = TypeVar("Seq")
Item = TypeVar("Item")

MIN_SCORE = -sys.maxsize - 1


class AlignmentMode(Enum):
    GLOBAL = auto()
    SEMI_GLOBAL = auto()
    HALF_LOCAL = auto()
    LOCAL = auto()


class PairwiseAlignmentAlgorithm(Generic[Seq, Item]):
    def __init__(
        self,
        scorer: PairwiseAlignmentScorer[Seq, Item],
        sequence1: Seq,
        sequence2: Seq,
        items_selector: Callable[[Seq], Tuple[Iterable[Item], int, int]],
        mode: AlignmentMode = AlignmentMode.GLOBAL,
        expansion_compression_enabled: bool = False,
        transposition_enabled: bool = False,
    ) -> None:
        self._scorer = scorer
        self._sequence1 = sequence1
        self._sequence2 = sequence2
        items1, self._start_index1, self._count1 = items_selector(sequence1)
        self._items1 = list(items1)
        items2, self._start_index2, self._count2 = items_selector(sequence2)
        self._items2 = list(items2)
        self._sim: List[List[int]] = [[0] * (self._count2 + 1) for _ in range(self._count1 + 1)]
        self._gap_penalty = self._scorer.get_gap_penalty(sequence1, sequence2)

        self.mode = mode
        self.expansion_compression_enabled = expansion_compression_enabled
        self.transposition_enabled = transposition_enabled
        self._best_raw_score = -1

    @property
    def best_raw_score(self) -> int:
        return self._best_raw_score

    def compute(self) -> None:
        max_score = MIN_SCORE

        if self.mode == AlignmentMode.GLOBAL or self.mode == AlignmentMode.HALF_LOCAL:
            for i in range(1, self._count1 + 1):
                self._sim[i][0] = (
                    self._sim[i - 1][0]
                    + self._gap_penalty
                    + self._scorer.get_deletion_score(self._sequence1, self._get1(i), self._sequence2, None)
                )

            for j in range(1, self._count2 + 1):
                self._sim[0][j] = (
                    self._sim[0][j - 1]
                    + self._gap_penalty
                    + self._scorer.get_insertion_score(self._sequence1, None, self._sequence2, self._get2(j))
                )

        for i in range(1, self._count1 + 1):
            for j in range(1, self._count2 + 1):
                m1 = (
                    self._sim[i - 1][j]
                    + self._gap_penalty
                    + self._scorer.get_deletion_score(self._sequence1, self._get1(i), self._sequence2, self._get2(j))
                )
                m2 = (
                    self._sim[i][j - 1]
                    + self._gap_penalty
                    + self._scorer.get_insertion_score(self._sequence1, self._get1(i), self._sequence2, self._get2(j))
                )
                m3 = self._sim[i - 1][j - 1] + self._scorer.get_substitution_score(
                    self._sequence1, self._get1(i), self._sequence2, self._get2(j)
                )
                m4 = (
                    MIN_SCORE
                    if not self.expansion_compression_enabled or j - 2 < 0
                    else self._sim[i - 1][j - 2]
                    + self._scorer.get_expansion_score(
                        self._sequence1, self._get1(i), self._sequence2, self._get2(j - 1), self._get2(j)
                    )
                )
                m5 = (
                    MIN_SCORE
                    if not self.expansion_compression_enabled or i - 2 < 0
                    else self._sim[i - 2][j - 1]
                    + self._scorer.get_compression_score(
                        self._sequence1, self._get1(i - 1), self._get1(i), self._sequence2, self._get2(j)
                    )
                )
                m6 = (
                    MIN_SCORE
                    if not self.transposition_enabled or i - 2 < 0 or j - 2 < 0
                    else self._sim[i - 2][j - 2]
                    + self._scorer.get_transposition_score(
                        self._sequence1,
                        self._get1(i - 1),
                        self._get1(i),
                        self._sequence2,
                        self._get2(j - 1),
                        self._get2(j),
                    )
                )

                if self.mode == AlignmentMode.LOCAL:
                    self._sim[i][j] = max([m1, m2, m3, m4, m5, m6, 0])
                else:
                    self._sim[i][j] = max([m1, m2, m3, m4, m5, m6])

                if self._sim[i][j] > max_score:
                    if self.mode == AlignmentMode.SEMI_GLOBAL:
                        if i == self._count1 or j == self._count2:
                            max_score = self._sim[i][j]
                    else:
                        max_score = self._sim[i][j]

        self._best_raw_score = (
            self._sim[-1][-1] if self.mode == AlignmentMode.GLOBAL or max_score == MIN_SCORE else max_score
        )

    def _get1(self, i: int) -> Item:
        if i == 0:
            raise ValueError("i is out of range.")
        return self._items1[self._start_index1 + i - 1]

    def _get2(self, j: int) -> Item:
        if j == 0:
            raise ValueError("j is out of range.")
        return self._items2[self._start_index2 + j - 1]

    def get_alignments(self, score_margin: float = 1.0) -> Iterable[Alignment[Seq, Item]]:
        threshold = int(self._best_raw_score * score_margin)

        if self.mode == AlignmentMode.GLOBAL:
            yield from self._get_alignments(self._count1, self._count2, threshold)
        elif self.mode == AlignmentMode.SEMI_GLOBAL:
            if self._count1 == 0 and self._count2 == 0:
                yield from self._get_alignments(0, 0, threshold)
            else:
                for i in range(1, self._count1 + 1):
                    yield from self._get_alignments(i, self._count2, threshold)

                for j in range(1, self._count2):
                    yield from self._get_alignments(self._count1, j, threshold)
        else:
            if self._count1 == 0 and self._count2 == 0:
                yield from self._get_alignments(0, 0, threshold)
            else:
                for i in range(1, self._count1 + 1):
                    for j in range(1, self._count2 + 1):
                        yield from self._get_alignments(i, j, threshold)

    def _get_alignments(self, i: int, j: int, threshold: int) -> Iterable[Alignment[Seq, Item]]:
        if self._sim[i][j] >= threshold:
            for aligned1, aligned2, start_index1, start_index2, score in self._retrieve(i, j, 0, threshold):
                end_index1 = self._start_index1 + i
                end_index2 = self._start_index2 + j
                normalized_score = self._calc_normalized_score(
                    start_index1, end_index1, start_index2, end_index2, score
                )
                yield Alignment[Seq, Item](
                    score,
                    normalized_score,
                    [
                        (
                            self._sequence1,
                            AlignmentCell(self._items1[:start_index1]),
                            aligned1,
                            AlignmentCell(self._items1[end_index1:]),
                        ),
                        (
                            self._sequence2,
                            AlignmentCell(self._items2[:start_index2]),
                            aligned2,
                            AlignmentCell(self._items2[end_index2:]),
                        ),
                    ],
                )

    def _calc_normalized_score(
        self, start_index1: int, end_index1: int, start_index2: int, end_index2: int, score: int
    ) -> float:
        max_score = max(
            self._calc_max_score1(start_index1, end_index1), self._calc_max_score2(start_index2, end_index2)
        )
        if max_score == 0:
            return 0
        return max(0, min(1, score / max_score))

    def _calc_max_score1(self, start_index: int, end_index: int) -> int:
        sum = 0
        for i in range(self._start_index1, self._count1):
            score = self._scorer.get_max_score1(self._sequence1, self._items1[i], self._sequence2)
            sum += score // 2 if i < start_index or i >= end_index else score
        return sum

    def _calc_max_score2(self, start_index: int, end_index: int) -> int:
        sum = 0
        for j in range(self._start_index2, self._count2):
            score = self._scorer.get_max_score2(self._sequence1, self._sequence2, self._items2[j])
            sum += score // 2 if j < start_index or j >= end_index else score
        return sum

    def _retrieve(
        self, i: int, j: int, score: int, threshold: int
    ) -> Iterable[Tuple[List[AlignmentCell[Item]], List[AlignmentCell[Item]], int, int, int]]:
        if (self.mode is AlignmentMode.LOCAL or self.mode is AlignmentMode.SEMI_GLOBAL) and (i == 0 or j == 0):
            yield self._create_alignment(i, j, score)
        elif i == 0 and j == 0:
            yield self._create_alignment(i, j, score)
        else:
            if i != 0 and j != 0:
                op_score = self._scorer.get_substitution_score(
                    self._sequence1, self._get1(i), self._sequence2, self._get2(j)
                )
                if self._sim[i - 1][j - 1] + op_score + score >= threshold:
                    for alignment in self._retrieve(i - 1, j - 1, score + op_score, threshold):
                        alignment[0].append(AlignmentCell([self._get1(i)]))
                        alignment[1].append(AlignmentCell([self._get2(j)]))
                        yield alignment

            if j != 0:
                op_score = self._gap_penalty + self._scorer.get_insertion_score(
                    self._sequence1, None if i == 0 else self._get1(i), self._sequence2, self._get2(j)
                )
                if i == 0 or self._sim[i][j - 1] + op_score + score >= threshold:
                    for alignment in self._retrieve(i, j - 1, score + op_score, threshold):
                        alignment[0].append(AlignmentCell[Item]())
                        alignment[1].append(AlignmentCell([self._get2(j)]))
                        yield alignment

            if self.expansion_compression_enabled and i != 0 and j - 2 >= 0:
                op_score = self._scorer.get_expansion_score(
                    self._sequence1, self._get1(i), self._sequence2, self._get2(j - 1), self._get2(j)
                )
                if self._sim[i - 1][j - 2] + op_score + score >= threshold:
                    for alignment in self._retrieve(i - 1, j - 2, score + op_score, threshold):
                        alignment[0].append(AlignmentCell([self._get1(i)]))
                        alignment[1].append(AlignmentCell([self._get2(j - 1), self._get2(j)]))
                        yield alignment

            if i != 0:
                op_score = self._gap_penalty + self._scorer.get_deletion_score(
                    self._sequence1, self._get1(i), self._sequence2, None if j == 0 else self._get2(j)
                )
                if j == 0 or self._sim[i - 1][j] + op_score + score >= threshold:
                    for alignment in self._retrieve(i - 1, j, score + op_score, threshold):
                        alignment[0].append(AlignmentCell([self._get1(i)]))
                        alignment[1].append(AlignmentCell[Item]())
                        yield alignment

            if self.expansion_compression_enabled and i - 2 >= 0 and j != 0:
                op_score = self._scorer.get_compression_score(
                    self._sequence1, self._get1(i - 1), self._get1(i), self._sequence2, self._get2(j)
                )
                if self._sim[i - 2][j - 1] + op_score + score >= threshold:
                    for alignment in self._retrieve(i - 2, j - 1, score + op_score, threshold):
                        alignment[0].append(AlignmentCell([self._get1(i - 1), self._get1(i)]))
                        alignment[1].append(AlignmentCell([self._get2(j)]))
                        yield alignment

            if self.transposition_enabled and i - 2 >= 0 and j - 2 >= 0:
                op_score = self._scorer.get_transposition_score(
                    self._sequence1, self._get1(i - 1), self._get1(i), self._sequence2, self._get2(j - 1), self._get2(j)
                )
                if self._sim[i - 2][j - 2] + op_score + score >= threshold:
                    for alignment in self._retrieve(i - 2, j - 2, score + op_score, threshold):
                        alignment[0].append(AlignmentCell([self._get1(i - 1)]))
                        alignment[0].append(AlignmentCell([self._get1(i)]))
                        alignment[1].append(AlignmentCell([self._get2(j)]))
                        alignment[1].append(AlignmentCell([self._get2(j - 1)]))
                        yield alignment

            if self.mode is AlignmentMode.LOCAL and self._sim[i][j] == 0:
                yield self._create_alignment(i, j, score)

    def _create_alignment(
        self, i: int, j: int, score: int
    ) -> Tuple[List[AlignmentCell[Item]], List[AlignmentCell[Item]], int, int, int]:
        return [], [], self._start_index1 + i, self._start_index2 + j, score

from typing import Callable, Collection, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sortedcontainers import SortedSet

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.parallel_text_segment import ParallelTextSegment
from .symmetrization_heuristic import SymmetrizationHeuristic


class WordAlignmentMatrix:
    @classmethod
    def from_parallel_text_segment(cls, segment: ParallelTextSegment) -> Optional["WordAlignmentMatrix"]:
        if segment.aligned_word_pairs is None:
            return None

        matrix = cls.from_word_pairs(len(segment.source_segment), len(segment.target_segment))
        for word_pair in segment.aligned_word_pairs:
            matrix[word_pair.source_index, word_pair.target_index] = True
        return matrix

    @classmethod
    def from_word_pairs(
        cls, row_count: int, column_count: int, set_values: Set[Tuple[int, int]] = set()
    ) -> "WordAlignmentMatrix":
        matrix = np.full((row_count, column_count), False)
        for i, j in set_values:
            matrix[i, j] = True
        return WordAlignmentMatrix(matrix)

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    @property
    def row_count(self) -> int:
        return self._matrix.shape[0]

    @property
    def column_count(self) -> int:
        return self._matrix.shape[1]

    def set_all(self, value: bool) -> None:
        self._matrix.fill(value)

    def __getitem__(self, key: Tuple[int, int]) -> bool:
        return self._matrix[key]

    def __setitem__(self, key: Tuple[int, int], value: bool) -> None:
        self._matrix[key] = value

    def is_row_aligned(self, i: int) -> bool:
        return bool(np.any(self._matrix[i, :]))

    def is_column_aligned(self, j: int) -> bool:
        return bool(np.any(self._matrix[:, j]))

    def get_row_aligned_indices(self, i: int) -> Iterable[int]:
        for j in range(self.column_count):
            if self._matrix[i, j]:
                yield j

    def get_column_aligned_indices(self, j: int) -> Iterable[int]:
        for i in range(self.row_count):
            if self._matrix[i, j]:
                yield i

    def is_diagonal_neighbor_aligned(self, i: int, j: int) -> bool:
        for di, dj in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            if self._get_safe(i + di, j + dj):
                return True
        return False

    def is_horizontal_neighbor_aligned(self, i: int, j: int) -> bool:
        for di, dj in [(0, 1), (0, -1)]:
            if self._get_safe(i + di, j + dj):
                return True
        return False

    def is_vertical_neighbor_aligned(self, i: int, j: int) -> bool:
        for di, dj in [(1, 0), (-1, 0)]:
            if self._get_safe(i + di, j + dj):
                return True
        return False

    def union_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        np.logical_or(self._matrix, other._matrix, out=self._matrix)

    def intersect_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        np.logical_and(self._matrix, other._matrix, out=self._matrix)

    def symmetrize_with(
        self, other: "WordAlignmentMatrix", heuristic: SymmetrizationHeuristic = SymmetrizationHeuristic.OCH
    ) -> None:
        if heuristic is SymmetrizationHeuristic.UNION:
            self.union_with(other)
        elif heuristic is SymmetrizationHeuristic.INTERSECTION:
            self.intersect_with(other)
        elif heuristic is SymmetrizationHeuristic.OCH:
            self.och_symmetrize_with(other)
        elif heuristic is SymmetrizationHeuristic.GROW:
            self.grow_symmetrize_with(other)
        elif heuristic is SymmetrizationHeuristic.GROW_DIAG:
            self.grow_diag_symmetrize_with(other)
        elif heuristic is SymmetrizationHeuristic.GROW_DIAG_FINAL:
            self.grow_diag_final_symmetrize_with(other)
        elif heuristic is SymmetrizationHeuristic.GROW_DIAG_FINAL_AND:
            self.grow_diag_final_and_symmetrize_with(other)

    def och_symmetrize_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        orig = self.copy()
        self.intersect_with(other)

        def is_block_neighbor_aligned(i: int, j: int) -> bool:
            return self.is_horizontal_neighbor_aligned(i, j) or self.is_vertical_neighbor_aligned(i, j)

        self._och_grow(is_block_neighbor_aligned, orig, other)

    def priority_symmetrize_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        def is_priority_block_neighbor_aligned(i: int, j: int) -> bool:
            return self.is_horizontal_neighbor_aligned(i, j) ^ self.is_vertical_neighbor_aligned(i, j)

        self._och_grow(is_priority_block_neighbor_aligned, self, other)

    def grow_symmetrize_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        orig = self.copy()
        self.intersect_with(other)

        def is_block_neighbor_aligned(i: int, j: int) -> bool:
            return self.is_horizontal_neighbor_aligned(i, j) or self.is_vertical_neighbor_aligned(i, j)

        self._koehn_grow(is_block_neighbor_aligned, orig, other)

    def grow_diag_symmetrize_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        orig = self.copy()
        self.intersect_with(other)

        def is_block_or_diag_neighbor_aligned(i: int, j: int) -> bool:
            return (
                self.is_horizontal_neighbor_aligned(i, j)
                or self.is_vertical_neighbor_aligned(i, j)
                or self.is_diagonal_neighbor_aligned(i, j)
            )

        self._koehn_grow(is_block_or_diag_neighbor_aligned, orig, other)

    def grow_diag_final_symmetrize_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        orig = self.copy()
        self.intersect_with(other)

        def is_block_or_diag_neighbor_aligned(i: int, j: int) -> bool:
            return (
                self.is_horizontal_neighbor_aligned(i, j)
                or self.is_vertical_neighbor_aligned(i, j)
                or self.is_diagonal_neighbor_aligned(i, j)
            )

        self._koehn_grow(is_block_or_diag_neighbor_aligned, orig, other)

        def is_one_or_both_unaligned(i: int, j: int) -> bool:
            return not self.is_row_aligned(i) or not self.is_column_aligned(j)

        self._final(is_one_or_both_unaligned, orig)
        self._final(is_one_or_both_unaligned, other)

    def grow_diag_final_and_symmetrize_with(self, other: "WordAlignmentMatrix") -> None:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            raise ValueError("The matrices are not the same size.")

        orig = self.copy()
        self.intersect_with(other)

        def is_block_or_diag_neighbor_aligned(i: int, j: int) -> bool:
            return (
                self.is_horizontal_neighbor_aligned(i, j)
                or self.is_vertical_neighbor_aligned(i, j)
                or self.is_diagonal_neighbor_aligned(i, j)
            )

        self._koehn_grow(is_block_or_diag_neighbor_aligned, orig, other)

        def is_both_unaligned(i: int, j: int) -> bool:
            return not self.is_row_aligned(i) and not self.is_column_aligned(j)

        self._final(is_both_unaligned, orig)
        self._final(is_both_unaligned, other)

    def transpose(self) -> None:
        self._matrix = np.transpose(self._matrix)

    def get_aligned_word_pairs(self) -> Collection[AlignedWordPair]:
        word_pairs, _, _ = self.get_asymmetric_alignments()
        return word_pairs

    def get_asymmetric_alignments(self) -> Tuple[Collection[AlignedWordPair], Sequence[int], Sequence[int]]:
        source = [0] * self.column_count
        target = [-2] * self.row_count
        word_pairs: List[AlignedWordPair] = []
        prev = -1
        for j in range(self.column_count):
            found = False
            for i in range(self.row_count):
                if self[i, j]:
                    if not found:
                        source[j] = i
                    if target[i] == -2:
                        target[i] = j
                    word_pairs.append(AlignedWordPair(i, j))
                    prev = i
                    found = True

            # unaligned indices
            if not found:
                source[j] = -1 if prev == -1 else self.row_count + prev

        # all remaining target indices are unaligned, so fill them in
        prev = -1
        for i in range(self.row_count):
            if target[i] == -2:
                target[i] = -1 if prev == -1 else self.column_count + prev
            else:
                prev = target[i]

        return word_pairs, source, target

    def to_giza_format(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> str:
        target_str = " ".join(target_segment)
        text = f"{target_str}\n"

        source_words = ["NULL"]
        source_words.extend(source_segment)

        i = 0
        for source_word in source_words:
            if i > 0:
                text += " "
            text += source_word + " ({ "
            for j in range(self.column_count):
                if i == 0:
                    if not self.is_column_aligned(j):
                        text += f"{j + 1} "
                elif self[i - 1, j]:
                    text += f"{j + 1} "

            text += "})"
            i += 1
        text += "\n"
        return text

    def copy(self) -> "WordAlignmentMatrix":
        return WordAlignmentMatrix(np.copy(self._matrix))

    def resize(self, row_count: int, column_count: int) -> None:
        if row_count == self.row_count and column_count == self.column_count:
            return
        new_matrix = np.full((row_count, column_count), False)
        rc = min(self.row_count, row_count)
        cc = min(self.column_count, column_count)
        new_matrix[:rc, :cc] = self._matrix[:rc, :cc]
        self._matrix = new_matrix

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordAlignmentMatrix):
            return NotImplemented

        return np.array_equal(self._matrix, other._matrix)

    def __repr__(self) -> str:
        return " ".join(str(wp) for wp in self.get_aligned_word_pairs())

    def _get_safe(self, i: int, j: int) -> bool:
        if i >= 0 and j >= 0 and i < self.row_count and j < self.column_count:
            return self[i, j]
        return False

    def _och_grow(
        self, grow_condition: Callable[[int, int], bool], orig: "WordAlignmentMatrix", other: "WordAlignmentMatrix"
    ) -> None:
        added = True
        while added:
            added = False
            for i in range(self.row_count):
                for j in range(self.column_count):
                    if (other[i, j] or orig[i, j]) and not self[i, j]:
                        if not self.is_row_aligned(i) and not self.is_column_aligned(j):
                            self[i, j] = True
                            added = True
                        elif grow_condition(i, j):
                            self[i, j] = True
                            added = True

    def _koehn_grow(
        self, grow_condition: Callable[[int, int], bool], orig: "WordAlignmentMatrix", other: "WordAlignmentMatrix"
    ) -> None:
        p = SortedSet()
        for i in range(self.row_count):
            for j in range(self.column_count):
                if (orig[i, j] or other[i, j]) and not self[i, j]:
                    p.add((i, j))

        keep_going = len(p) > 0
        while keep_going:
            keep_going = False
            added = SortedSet()
            for i, j in p:
                if (not self.is_row_aligned(i) or not self.is_column_aligned(j)) and grow_condition(i, j):
                    self[i, j] = True
                    added.add((i, j))
                    keep_going = True
            p.difference_update(added)

    def _final(self, pred: Callable[[int, int], bool], adds: "WordAlignmentMatrix") -> None:
        for i in range(self.row_count):
            for j in range(self.column_count):
                if adds[i, j] and not self[i, j] and pred(i, j):
                    self[i, j] = True

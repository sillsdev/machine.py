from abc import ABC, abstractmethod
from typing import Sequence

from ..corpora.parallel_text_row import ParallelTextRow
from .word_alignment_matrix import WordAlignmentMatrix


class WordAligner(ABC):
    @abstractmethod
    def align(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        ...

    @abstractmethod
    def align_batch(self, segments: Sequence[Sequence[Sequence[str]]]) -> Sequence[WordAlignmentMatrix]:
        ...

    def align_parallel_text_row(self, row: ParallelTextRow) -> WordAlignmentMatrix:
        alignment = self.align(row.source_segment, row.target_segment)
        known_alignment = WordAlignmentMatrix.from_parallel_text_row(row)
        if known_alignment is not None:
            known_alignment.priority_symmetrize_with(alignment)
            alignment = known_alignment
        return alignment

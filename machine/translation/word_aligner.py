from abc import ABC, abstractmethod
from typing import Optional, Sequence

from .word_alignment_matrix import WordAlignmentMatrix


class WordAligner(ABC):
    @abstractmethod
    def get_best_alignment(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        ...

    @abstractmethod
    def get_best_alignments(
        self, source_segments: Sequence[Sequence[str]], target_segments: Sequence[Sequence[str]]
    ) -> Sequence[WordAlignmentMatrix]:
        ...

    def get_best_alignment_from_known(
        self,
        source_segment: Sequence[str],
        target_segment: Sequence[str],
        known_alignment: Optional[WordAlignmentMatrix],
    ) -> WordAlignmentMatrix:
        estimated_alignment = self.get_best_alignment(source_segment, target_segment)
        alignment = estimated_alignment
        if known_alignment is not None:
            alignment = known_alignment.copy()
            alignment.priority_symmetrize_with(estimated_alignment)
        return alignment

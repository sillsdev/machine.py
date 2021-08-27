from typing import Sequence

from .symmetrization_heuristic import SymmetrizationHeuristic
from .word_aligner import WordAligner
from .word_alignment_matrix import WordAlignmentMatrix


class SymmetrizedWordAligner(WordAligner):
    def __init__(self, src_trg_aligner: WordAligner, trg_src_aligner: WordAligner) -> None:
        self._src_trg_aligner = src_trg_aligner
        self._trg_src_aligner = trg_src_aligner
        self._heuristic = SymmetrizationHeuristic.OCH

    @property
    def heuristic(self) -> SymmetrizationHeuristic:
        return self._heuristic

    @heuristic.setter
    def heuristic(self, value: SymmetrizationHeuristic) -> None:
        self._heuristic = value

    def get_best_alignment(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        matrix = self._src_trg_aligner.get_best_alignment(source_segment, target_segment)
        if self.heuristic is not SymmetrizationHeuristic.NONE:
            inv_matrix = self._trg_src_aligner.get_best_alignment(target_segment, source_segment)
            inv_matrix.transpose()
            matrix.symmetrize_with(inv_matrix, self.heuristic)
        return matrix

    def get_best_alignments(
        self, source_segments: Sequence[Sequence[str]], target_segments: Sequence[Sequence[str]]
    ) -> Sequence[WordAlignmentMatrix]:
        matrices = self._src_trg_aligner.get_best_alignments(source_segments, target_segments)
        if self.heuristic is not SymmetrizationHeuristic.NONE:
            inv_matrices = self._trg_src_aligner.get_best_alignments(target_segments, source_segments)
            for i in range(len(matrices)):
                inv_matrices[i].transpose()
                matrices[i].symmetrize_with(inv_matrices[i], self.heuristic)
        return matrices

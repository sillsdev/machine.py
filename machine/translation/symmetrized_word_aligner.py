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

    def align(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        matrix = self._src_trg_aligner.align(source_segment, target_segment)
        if self.heuristic is not SymmetrizationHeuristic.NONE:
            inv_matrix = self._trg_src_aligner.align(target_segment, source_segment)
            inv_matrix.transpose()
            matrix.symmetrize_with(inv_matrix, self.heuristic)
        return matrix

    def align_batch(self, segments: Sequence[Sequence[Sequence[str]]]) -> Sequence[WordAlignmentMatrix]:
        if self.heuristic is SymmetrizationHeuristic.NONE:
            return self._src_trg_aligner.align_batch(segments)
        else:
            results = self._src_trg_aligner.align_batch(segments)
            inv_results = self._trg_src_aligner.align_batch(
                [(target_segment, source_segment) for source_segment, target_segment in segments]
            )
            for matrix, inv_matrix in zip(results, inv_results):
                inv_matrix.transpose()
                matrix.symmetrize_with(inv_matrix, self.heuristic)
            return results

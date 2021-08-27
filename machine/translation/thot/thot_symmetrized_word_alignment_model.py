from typing import Sequence

import thot.alignment as ta

from ..symmetrization_heuristic import SymmetrizationHeuristic
from ..symmetrized_word_alignment_model import SymmetrizedWordAlignmentModel
from ..word_alignment_matrix import WordAlignmentMatrix
from .thot_word_alignment_model import ThotWordAlignmentModel


class ThotSymmetrizedWordAlignmentModel(SymmetrizedWordAlignmentModel):
    def __init__(
        self, direct_word_alignment_model: ThotWordAlignmentModel, inverse_word_alignment_model: ThotWordAlignmentModel
    ) -> None:
        super().__init__(direct_word_alignment_model, inverse_word_alignment_model)
        self._aligner = ta.SymmetrizedAligner(
            direct_word_alignment_model.thot_model, inverse_word_alignment_model.thot_model
        )
        self._aligner.heuristic = _convert_heuristic(self._heuristic)

    @property
    def heuristic(self) -> SymmetrizationHeuristic:
        return self._heuristic

    @heuristic.setter
    def heuristic(self, value: SymmetrizationHeuristic) -> None:
        self._heuristic = value
        self._aligner.heuristic = _convert_heuristic(self._heuristic)

    def get_best_alignment(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        _, matrix = self._aligner.get_best_alignment(source_segment, target_segment)
        return WordAlignmentMatrix(matrix.to_numpy())

    def get_best_alignments(
        self, source_segments: Sequence[Sequence[str]], target_segments: Sequence[Sequence[str]]
    ) -> Sequence[WordAlignmentMatrix]:
        if len(source_segments) != len(target_segments):
            raise ValueError("The number of source and target segments must be equal.")
        return [
            WordAlignmentMatrix(matrix.to_numpy())
            for _, matrix in self._aligner.get_best_alignments(source_segments, target_segments)
        ]


def _convert_heuristic(machine_heuristic: SymmetrizationHeuristic) -> ta.SymmetrizationHeuristic:
    if machine_heuristic is SymmetrizationHeuristic.NONE:
        return ta.SymmetrizationHeuristic.NONE
    if machine_heuristic is SymmetrizationHeuristic.UNION:
        return ta.SymmetrizationHeuristic.UNION
    if machine_heuristic is SymmetrizationHeuristic.INTERSECTION:
        return ta.SymmetrizationHeuristic.INTERSECTION
    if machine_heuristic is SymmetrizationHeuristic.OCH:
        return ta.SymmetrizationHeuristic.OCH
    if machine_heuristic is SymmetrizationHeuristic.GROW:
        return ta.SymmetrizationHeuristic.GROW
    if machine_heuristic is SymmetrizationHeuristic.GROW_DIAG:
        return ta.SymmetrizationHeuristic.GROW_DIAG
    if machine_heuristic is SymmetrizationHeuristic.GROW_DIAG_FINAL:
        return ta.SymmetrizationHeuristic.GROW_DIAG_FINAL
    if machine_heuristic is SymmetrizationHeuristic.GROW_DIAG_FINAL_AND:
        return ta.SymmetrizationHeuristic.GROW_DIAG_FINAL_AND
    return ta.SymmetrizationHeuristic.NONE

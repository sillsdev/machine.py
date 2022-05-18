from typing import Iterable, Sequence, Tuple

from ..corpora.corpora_utils import batch
from .symmetrization_heuristic import SymmetrizationHeuristic
from .word_aligner import WordAligner
from .word_alignment_matrix import WordAlignmentMatrix


class SymmetrizedWordAligner(WordAligner):
    def __init__(self, src_trg_aligner: WordAligner, trg_src_aligner: WordAligner) -> None:
        self._src_trg_aligner = src_trg_aligner
        self._trg_src_aligner = trg_src_aligner
        self._heuristic = SymmetrizationHeuristic.OCH
        self.batch_size = 1024

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

    def get_best_alignment_batch(
        self, segments: Iterable[Tuple[Sequence[str], Sequence[str]]]
    ) -> Iterable[Tuple[Sequence[str], Sequence[str], WordAlignmentMatrix]]:
        if self.heuristic is SymmetrizationHeuristic.NONE:
            yield from self._src_trg_aligner.get_best_alignment_batch(segments)
        else:
            for segments_batch in batch(segments, self.batch_size):
                results = self._src_trg_aligner.get_best_alignment_batch(segments_batch)
                inv_results = self._trg_src_aligner.get_best_alignment_batch(
                    (target_segment, source_segment) for source_segment, target_segment in segments_batch
                )
                for (source_segment, target_segment, matrix), (_, _, inv_matrix) in zip(results, inv_results):
                    inv_matrix.transpose()
                    matrix.symmetrize_with(inv_matrix, self.heuristic)
                    yield source_segment, target_segment, matrix

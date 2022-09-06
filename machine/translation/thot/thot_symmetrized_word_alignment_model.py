from typing import Iterable, Optional, Sequence, Tuple, cast

import thot.alignment as ta

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ..symmetrization_heuristic import SymmetrizationHeuristic
from ..symmetrized_word_alignment_model import SymmetrizedWordAlignmentModel
from ..symmetrized_word_alignment_model_trainer import SymmetrizedWordAlignmentModelTrainer
from ..trainer import Trainer
from ..word_alignment_matrix import WordAlignmentMatrix
from .thot_utils import batch
from .thot_word_alignment_model import ThotWordAlignmentModel


class ThotSymmetrizedWordAlignmentModel(SymmetrizedWordAlignmentModel):
    def __init__(
        self, direct_word_alignment_model: ThotWordAlignmentModel, inverse_word_alignment_model: ThotWordAlignmentModel
    ) -> None:
        super().__init__(direct_word_alignment_model, inverse_word_alignment_model)
        self._reset_aligner()

    @property
    def heuristic(self) -> SymmetrizationHeuristic:
        return self._heuristic

    @property
    def direct_word_alignment_model(self) -> ThotWordAlignmentModel:
        return cast(ThotWordAlignmentModel, self._direct_word_alignment_model)

    @property
    def inverse_word_alignment_model(self) -> ThotWordAlignmentModel:
        return cast(ThotWordAlignmentModel, self._inverse_word_alignment_model)

    @heuristic.setter
    def heuristic(self, value: SymmetrizationHeuristic) -> None:
        self._heuristic = value
        self._aligner.heuristic = _convert_heuristic(self._heuristic)

    def get_best_alignment(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        _, matrix = self._aligner.get_best_alignment(source_segment, target_segment)
        return WordAlignmentMatrix(matrix.to_numpy())

    def get_best_alignment_batch(
        self, segments: Iterable[Tuple[Sequence[str], Sequence[str]]], batch_size: Optional[int] = None
    ) -> Iterable[Tuple[Sequence[str], Sequence[str], WordAlignmentMatrix]]:
        for source_segments, target_segments in batch(segments, self.batch_size if batch_size is None else batch_size):
            results = self._aligner.get_best_alignments(source_segments, target_segments)
            for source_segment, target_segment, (_, matrix) in zip(source_segments, target_segments, results):
                yield source_segment, target_segment, WordAlignmentMatrix(matrix.to_numpy())

    def create_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        direct_trainer = self._direct_word_alignment_model.create_trainer(corpus)
        inverse_trainer = self._inverse_word_alignment_model.create_trainer(corpus.invert())

        return _Trainer(self, direct_trainer, inverse_trainer)

    def _reset_aligner(self) -> None:
        self._aligner = ta.SymmetrizedAligner(
            self.direct_word_alignment_model.thot_model, self.inverse_word_alignment_model.thot_model
        )
        self._aligner.heuristic = _convert_heuristic(self._heuristic)


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


class _Trainer(SymmetrizedWordAlignmentModelTrainer):
    def __init__(
        self, model: ThotSymmetrizedWordAlignmentModel, direct_trainer: Trainer, inverse_trainer: Trainer
    ) -> None:
        super().__init__(direct_trainer, inverse_trainer)
        self._model = model

    def save(self) -> None:
        super().save()
        self._model._reset_aligner()

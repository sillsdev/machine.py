from typing import cast

import thot.alignment as ta

from ..ibm2_word_alignment_model import Ibm2WordAlignmentModel
from .thot_ibm1_word_alignment_model import ThotIbm1WordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotIbm2WordAlignmentModel(ThotIbm1WordAlignmentModel, Ibm2WordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.IBM2

    def get_alignment_score(
        self,
        source_length: int,
        prev_source_index: int,
        source_index: int,
        target_length: int,
        prev_target_index: int,
        target_index: int,
    ) -> float:
        return self.get_alignment_probability(source_length, source_index, target_length, target_index)

    def get_alignment_probability(
        self, source_length: int, source_index: int, target_length: int, target_index: int
    ) -> float:
        # add 1 to convert the specified indices to Thot position indices, which are 1-based
        return self._ibm2_model.alignment_prob(target_index + 1, source_length, target_length, source_index + 1)

    @property
    def _ibm2_model(self) -> ta.Ibm2AlignmentModel:
        return cast(ta.Ibm2AlignmentModel, self._model)

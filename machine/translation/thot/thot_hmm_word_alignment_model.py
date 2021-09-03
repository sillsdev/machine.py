from typing import cast

import thot.alignment as ta

from ..hmm_word_alignment_model import HmmWordAlignmentModel
from .thot_ibm1_word_alignment_model import ThotIbm1WordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotHmmWordAlignmentModel(ThotIbm1WordAlignmentModel, HmmWordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.HMM

    def get_alignment_score(
        self,
        source_length: int,
        prev_source_index: int,
        source_index: int,
        target_length: int,
        prev_target_index: int,
        target_index: int,
    ) -> float:
        return self.get_alignment_probability(source_length, prev_source_index, source_index)

    def get_alignment_probability(self, source_length: int, prev_source_index: int, source_index: int) -> float:
        # add 1 to convert the specified indices to Thot position indices, which are 1-based
        return self._hmm_model.hmm_alignment_prob(prev_source_index + 1, source_length, source_index + 1)

    @property
    def _hmm_model(self) -> ta.HmmAlignmentModel:
        return cast(ta.HmmAlignmentModel, self._model)

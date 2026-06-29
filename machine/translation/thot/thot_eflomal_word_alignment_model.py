from __future__ import annotations

from typing import cast

import thot.alignment as ta

from .thot_hmm_word_alignment_model_base import ThotHmmWordAlignmentModelBase
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotEflomalWordAlignmentModel(ThotHmmWordAlignmentModelBase):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.EFLOMAL

    @property
    def thot_model(self) -> ta.EflomalAlignmentModel:
        return cast(ta.EflomalAlignmentModel, self._model)

    def get_alignment_log_probability(self, source_length: int, prev_source_index: int, source_index: int) -> float:
        # add 1 to convert the specified indices to Thot position indices, which are 1-based
        return self.thot_model.hmm_alignment_log_prob(prev_source_index + 1, source_length, source_index + 1)

    def __enter__(self) -> ThotEflomalWordAlignmentModel:
        return self

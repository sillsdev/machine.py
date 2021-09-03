from typing import Optional, cast

import thot.alignment as ta

from ...utils.typeshed import StrPath
from ..ibm2_word_alignment_model import Ibm2WordAlignmentModel
from .thot_word_alignment_model import ThotWordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotFastAlignWordAlignmentModel(ThotWordAlignmentModel, Ibm2WordAlignmentModel):
    def __init__(self, prefix_filename: Optional[StrPath] = None, create_new: bool = False) -> None:
        super().__init__(prefix_filename, create_new)
        self.training_iteration_count = 4

    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.FAST_ALIGN

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
        return self._fa_model.alignment_prob(target_index + 1, source_length, target_length, source_index + 1)

    @property
    def _fa_model(self) -> ta.FastAlignModel:
        return cast(ta.FastAlignModel, self._model)

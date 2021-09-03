from .thot_word_alignment_model import ThotWordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotIbm1WordAlignmentModel(ThotWordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.IBM1

    def get_alignment_score(
        self,
        source_length: int,
        prev_source_index: int,
        source_index: int,
        target_length: int,
        prev_target_index: int,
        target_index: int,
    ) -> float:
        return 1.0 / (source_length + 1)

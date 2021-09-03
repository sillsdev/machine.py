from .thot_ibm3_word_alignment_model import ThotIbm3WordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotIbm4WordAlignmentModel(ThotIbm3WordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.IBM4

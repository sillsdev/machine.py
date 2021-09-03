from .thot_ibm2_word_alignment_model import ThotIbm2WordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotIbm3WordAlignmentModel(ThotIbm2WordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.IBM3

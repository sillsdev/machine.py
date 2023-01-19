from typing import Union

from .thot_fast_align_word_alignment_model import ThotFastAlignWordAlignmentModel
from .thot_hmm_word_alignment_model import ThotHmmWordAlignmentModel
from .thot_ibm1_word_alignment_model import ThotIbm1WordAlignmentModel
from .thot_ibm2_word_alignment_model import ThotIbm2WordAlignmentModel
from .thot_ibm3_word_alignment_model import ThotIbm3WordAlignmentModel
from .thot_ibm4_word_alignment_model import ThotIbm4WordAlignmentModel
from .thot_symmetrized_word_alignment_model import ThotSymmetrizedWordAlignmentModel
from .thot_word_alignment_model import ThotWordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


def create_thot_word_alignment_model(type: Union[str, int]) -> ThotWordAlignmentModel:
    if isinstance(type, str):
        type = ThotWordAlignmentModelType[type.upper()]
    if type == ThotWordAlignmentModelType.FAST_ALIGN:
        return ThotFastAlignWordAlignmentModel()
    if type == ThotWordAlignmentModelType.IBM1:
        return ThotIbm1WordAlignmentModel()
    if type == ThotWordAlignmentModelType.IBM2:
        return ThotIbm2WordAlignmentModel()
    if type == ThotWordAlignmentModelType.HMM:
        return ThotHmmWordAlignmentModel()
    if type == ThotWordAlignmentModelType.IBM3:
        return ThotIbm3WordAlignmentModel()
    if type == ThotWordAlignmentModelType.IBM4:
        return ThotIbm4WordAlignmentModel()
    raise ValueError("The word alignment model type is unknown.")


def create_thot_symmetrized_word_alignment_model(type: Union[int, str]) -> ThotSymmetrizedWordAlignmentModel:
    direct_model = create_thot_word_alignment_model(type)
    inverse_model = create_thot_word_alignment_model(type)
    return ThotSymmetrizedWordAlignmentModel(direct_model, inverse_model)

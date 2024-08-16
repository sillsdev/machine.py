from typing import Optional, Union

from ...utils.typeshed import StrPath
from .thot_fast_align_word_alignment_model import ThotFastAlignWordAlignmentModel
from .thot_hmm_word_alignment_model import ThotHmmWordAlignmentModel
from .thot_ibm1_word_alignment_model import ThotIbm1WordAlignmentModel
from .thot_ibm2_word_alignment_model import ThotIbm2WordAlignmentModel
from .thot_ibm3_word_alignment_model import ThotIbm3WordAlignmentModel
from .thot_ibm4_word_alignment_model import ThotIbm4WordAlignmentModel
from .thot_symmetrized_word_alignment_model import ThotSymmetrizedWordAlignmentModel
from .thot_word_alignment_model import ThotWordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


def create_thot_word_alignment_model(
    type: Union[str, int], prefix_filename: Optional[StrPath] = None
) -> ThotWordAlignmentModel:
    if isinstance(type, str):
        type = ThotWordAlignmentModelType[type.upper()]
    if type == ThotWordAlignmentModelType.FAST_ALIGN:
        return ThotFastAlignWordAlignmentModel(prefix_filename)
    if type == ThotWordAlignmentModelType.IBM1:
        return ThotIbm1WordAlignmentModel(prefix_filename)
    if type == ThotWordAlignmentModelType.IBM2:
        return ThotIbm2WordAlignmentModel(prefix_filename)
    if type == ThotWordAlignmentModelType.HMM:
        return ThotHmmWordAlignmentModel(prefix_filename)
    if type == ThotWordAlignmentModelType.IBM3:
        return ThotIbm3WordAlignmentModel(prefix_filename)
    if type == ThotWordAlignmentModelType.IBM4:
        return ThotIbm4WordAlignmentModel(prefix_filename)
    raise ValueError("The word alignment model type is unknown.")


def create_thot_symmetrized_word_alignment_model(
    type: Union[int, str],
    direct_prefix_filename: Optional[StrPath] = None,
    inverse_prefix_filename: Optional[StrPath] = None,
) -> ThotSymmetrizedWordAlignmentModel:
    direct_model = create_thot_word_alignment_model(type, direct_prefix_filename)
    inverse_model = create_thot_word_alignment_model(type, inverse_prefix_filename)
    return ThotSymmetrizedWordAlignmentModel(direct_model, inverse_model)

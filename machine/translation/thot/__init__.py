from ...utils.packages import is_thot_available

if not is_thot_available():
    raise RuntimeError('sil-thot is not installed. Install sil-machine with the "thot" extra.')

from .thot_fast_align_word_alignment_model import ThotFastAlignWordAlignmentModel
from .thot_hmm_word_alignment_model import ThotHmmWordAlignmentModel
from .thot_ibm1_word_alignment_model import ThotIbm1WordAlignmentModel
from .thot_ibm2_word_alignment_model import ThotIbm2WordAlignmentModel
from .thot_ibm3_word_alignment_model import ThotIbm3WordAlignmentModel
from .thot_ibm4_word_alignment_model import ThotIbm4WordAlignmentModel
from .thot_smt_model import ThotSmtModel
from .thot_smt_model_trainer import ThotSmtModelTrainer
from .thot_smt_parameters import ThotSmtParameters
from .thot_symmetrized_word_alignment_model import ThotSymmetrizedWordAlignmentModel
from .thot_word_alignment_model import ThotWordAlignmentModel
from .thot_word_alignment_model_trainer import ThotWordAlignmentModelTrainer
from .thot_word_alignment_model_type import ThotWordAlignmentModelType
from .thot_word_alignment_model_utils import (
    create_thot_symmetrized_word_alignment_model,
    create_thot_word_alignment_model,
)
from .thot_word_alignment_parameters import ThotWordAlignmentParameters

__all__ = [
    "create_thot_symmetrized_word_alignment_model",
    "create_thot_word_alignment_model",
    "ThotFastAlignWordAlignmentModel",
    "ThotHmmWordAlignmentModel",
    "ThotIbm1WordAlignmentModel",
    "ThotIbm2WordAlignmentModel",
    "ThotIbm3WordAlignmentModel",
    "ThotIbm4WordAlignmentModel",
    "ThotSmtModel",
    "ThotSmtModelTrainer",
    "ThotSmtParameters",
    "ThotSymmetrizedWordAlignmentModel",
    "ThotWordAlignmentModel",
    "ThotWordAlignmentModelTrainer",
    "ThotWordAlignmentModelType",
    "ThotWordAlignmentParameters",
]

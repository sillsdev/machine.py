from dataclasses import dataclass, field
from typing import Dict, Optional

from .thot_word_alignment_model_type import ThotWordAlignmentModelType


@dataclass
class ThotWordAlignmentParameters:
    ibm1_iteration_count: Optional[int] = None
    ibm2_iteration_count: Optional[int] = None
    hmm_iteration_count: Optional[int] = None
    ibm3_iteration_count: Optional[int] = None
    ibm4_iteration_count: Optional[int] = None
    fast_align_iteration_count: Optional[int] = None
    variational_bayes: Optional[bool] = None
    fast_align_p0: Optional[float] = None
    hmm_p0: Optional[float] = None
    hmm_lexical_smoothing_factor: Optional[float] = None
    hmm_alignment_smoothing_factor: Optional[float] = None
    ibm3_fertility_smoothing_factor: Optional[float] = None
    ibm3_count_threshold: Optional[float] = None
    ibm4_distortion_smoothing_factor: Optional[float] = None
    source_word_classes: Dict[str, str] = field(default_factory=dict)
    target_word_classes: Dict[str, str] = field(default_factory=dict)

    def get_ibm1_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        return _get_ibm_iteration_count(model_type, ThotWordAlignmentModelType.IBM1, self.ibm1_iteration_count)

    def get_ibm2_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        return _get_ibm_iteration_count(
            model_type, ThotWordAlignmentModelType.IBM2, self.ibm2_iteration_count, default_init_iteration_count=0
        )

    def get_hmm_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        if (
            model_type is not ThotWordAlignmentModelType.HMM
            and self.ibm2_iteration_count is not None
            and self.ibm2_iteration_count > 0
        ):
            return 0
        return _get_ibm_iteration_count(model_type, ThotWordAlignmentModelType.HMM, self.hmm_iteration_count)

    def get_ibm3_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        return _get_ibm_iteration_count(model_type, ThotWordAlignmentModelType.IBM3, self.ibm3_iteration_count)

    def get_ibm4_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        return _get_ibm_iteration_count(model_type, ThotWordAlignmentModelType.IBM4, self.ibm4_iteration_count)

    def get_fast_align_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        if model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            return 4 if self.fast_align_iteration_count is None else self.fast_align_iteration_count
        return 0

    def get_variational_bayes(self, model_type: ThotWordAlignmentModelType) -> bool:
        if self.variational_bayes is None:
            return model_type is ThotWordAlignmentModelType.FAST_ALIGN
        return self.variational_bayes


def _get_ibm_iteration_count(
    model_type: ThotWordAlignmentModelType,
    iteration_count_model_type: ThotWordAlignmentModelType,
    iteration_count: Optional[int],
    default_init_iteration_count: int = 5,
) -> int:
    if model_type < iteration_count_model_type:
        return 0
    if iteration_count is None:
        return 4 if model_type is iteration_count_model_type else default_init_iteration_count
    return iteration_count

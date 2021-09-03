from dataclasses import dataclass, field
from typing import Dict, Optional

from .thot_word_alignment_model_type import ThotWordAlignmentModelType


@dataclass
class ThotWordAlignmentModelParameters:
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
        if model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            return 0
        return 5 if self.ibm1_iteration_count is None else self.ibm1_iteration_count

    def get_ibm2_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        if model_type in {ThotWordAlignmentModelType.IBM1, ThotWordAlignmentModelType.FAST_ALIGN}:
            return 0
        if model_type is ThotWordAlignmentModelType.IBM2:
            return 5 if self.ibm2_iteration_count is None else self.ibm2_iteration_count
        return 0 if self.ibm2_iteration_count is None else self.ibm2_iteration_count

    def get_hmm_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        if (
            model_type
            in {
                ThotWordAlignmentModelType.IBM1,
                ThotWordAlignmentModelType.IBM2,
                ThotWordAlignmentModelType.FAST_ALIGN,
            }
            or (self.ibm2_iteration_count is not None and self.ibm2_iteration_count > 0)
        ):
            return 0
        return 5 if self.hmm_iteration_count is None else self.hmm_iteration_count

    def get_ibm3_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        if model_type in {
            ThotWordAlignmentModelType.IBM1,
            ThotWordAlignmentModelType.FAST_ALIGN,
            ThotWordAlignmentModelType.IBM2,
            ThotWordAlignmentModelType.HMM,
        }:
            return 0
        return 5 if self.ibm3_iteration_count is None else self.ibm3_iteration_count

    def get_ibm4_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        if model_type in {
            ThotWordAlignmentModelType.IBM1,
            ThotWordAlignmentModelType.FAST_ALIGN,
            ThotWordAlignmentModelType.IBM2,
            ThotWordAlignmentModelType.HMM,
            ThotWordAlignmentModelType.IBM3,
        }:
            return 0
        return 5 if self.ibm4_iteration_count is None else self.ibm4_iteration_count

    def get_fast_align_iteration_count(self, model_type: ThotWordAlignmentModelType) -> int:
        if model_type is not ThotWordAlignmentModelType.FAST_ALIGN:
            return 0
        return 4 if self.fast_align_iteration_count is None else self.fast_align_iteration_count

    def get_variational_bayes(self, model_type: ThotWordAlignmentModelType) -> bool:
        if self.variational_bayes is None:
            return model_type is ThotWordAlignmentModelType.FAST_ALIGN
        return self.variational_bayes

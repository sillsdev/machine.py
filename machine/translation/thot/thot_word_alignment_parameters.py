from dataclasses import dataclass, field
from typing import Dict, Optional

from .thot_word_alignment_model_type import ThotWordAlignmentModelType

_DEFAULT_EFLOMAL_IBM1_ITERATION_COUNT = 8
_DEFAULT_EFLOMAL_HMM_ITERATION_COUNT = 8
_DEFAULT_EFLOMAL_FERTILITY_ITERATION_COUNT = 32


@dataclass
class ThotWordAlignmentParameters:
    ibm1_iteration_count: Optional[int] = None
    ibm2_iteration_count: Optional[int] = None
    hmm_iteration_count: Optional[int] = None
    ibm3_iteration_count: Optional[int] = None
    ibm4_iteration_count: Optional[int] = None
    fast_align_iteration_count: Optional[int] = None
    # Number of independent Gibbs chains trained in parallel. Marginals are summed across
    # chains at decode time (eflomal's n_samplers scheme). Default 3.
    eflomal_num_samplers: Optional[int] = None
    # Trains the Gibbs sampler chains serially rather than across threads, so a fixed seed
    # produces a reproducible model at the cost of parallelism. Default False.
    eflomal_deterministic: Optional[bool] = None
    # Random seed for the Gibbs samplers. Combine with eflomal_deterministic for fully
    # reproducible training. Default 1351155463.
    eflomal_seed: Optional[int] = None
    # When True, the lexical model uses the plain denominator 1/N(e) instead of the
    # Dirichlet-smoothed 1/(N(e) + alpha_lex * |V|). Default True.
    eflomal_lex_norm: Optional[bool] = None
    # Lexical Dirichlet prior for source words. Matches eflomal LEX_ALPHA. Default 0.001.
    eflomal_lex_alpha: Optional[float] = None
    # Jump distribution Dirichlet prior. Matches eflomal JUMP_ALPHA. Default 0.5.
    eflomal_jump_alpha: Optional[float] = None
    # Fertility distribution Dirichlet prior. Matches eflomal FERT_ALPHA. Default 0.5.
    eflomal_fertility_alpha: Optional[float] = None
    # Fixed probability of aligning a target token to the NULL source word (IBM1 mixing
    # weight). Not a Dirichlet prior; controls the null/non-null split. Default 0.2.
    eflomal_p0: Optional[float] = None
    # Half-width of the jump distribution window. Offsets beyond +/-jump_window are clamped.
    # Roughly corresponds to eflomal JUMP_MAX_EST. Default 100.
    eflomal_jump_window: Optional[int] = None
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
            model_type,
            ThotWordAlignmentModelType.IBM2,
            self.ibm2_iteration_count,
            default_init_iteration_count=0,
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

    # Eflomal runs its IBM1->HMM->fertility cascade internally as a single model, reusing the
    # IBM1/HMM/IBM3 iteration counts for its three stages (IBM3 drives the fertility stage).
    # When none of them are specified, the model derives the schedule automatically from the
    # corpus size, so no explicit schedule should be set.
    @property
    def is_eflomal_schedule_specified(self) -> bool:
        return (
            self.ibm1_iteration_count is not None
            or self.hmm_iteration_count is not None
            or self.ibm3_iteration_count is not None
        )

    def get_eflomal_ibm1_iteration_count(self) -> int:
        return _DEFAULT_EFLOMAL_IBM1_ITERATION_COUNT if self.ibm1_iteration_count is None else self.ibm1_iteration_count

    def get_eflomal_hmm_iteration_count(self) -> int:
        return _DEFAULT_EFLOMAL_HMM_ITERATION_COUNT if self.hmm_iteration_count is None else self.hmm_iteration_count

    def get_eflomal_fertility_iteration_count(self) -> int:
        return (
            _DEFAULT_EFLOMAL_FERTILITY_ITERATION_COUNT
            if self.ibm3_iteration_count is None
            else self.ibm3_iteration_count
        )

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

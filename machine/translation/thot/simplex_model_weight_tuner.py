from typing import Callable, Optional, Sequence

import numpy as np
import thot.translation as tt

from ...optimization import NelderMeadSimplex
from ...utils.progress_status import ProgressStatus
from ..evaluation import compute_bleu
from ..trainer import TrainStats
from .parameter_tuner import ParameterTuner
from .thot_smt_parameters import ThotSmtParameters
from .thot_utils import escape_tokens, load_smt_decoder, load_smt_model
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class SimplexModelWeightTuner(ParameterTuner):
    def __init__(
        self,
        word_alignment_model_type: ThotWordAlignmentModelType,
        convergence_tolerance: float = 0.001,
        max_function_evaluations: int = 100,
        max_progress_function_evaluations: int = 70,
    ) -> None:
        self._word_alignment_model_type = word_alignment_model_type
        self.convergence_tolerance = convergence_tolerance
        self.max_function_evaluations = max_function_evaluations
        self.max_progress_function_evaluations = max_progress_function_evaluations

    def tune(
        self,
        parameters: ThotSmtParameters,
        tune_source_corpus: Sequence[Sequence[str]],
        tune_target_corpus: Sequence[Sequence[str]],
        stats: TrainStats,
        progress: Callable[[ProgressStatus], None],
    ) -> ThotSmtParameters:
        sent_len_weight = parameters.model_weights[7]

        def evaluate(weights: np.ndarray, eval_count: int) -> float:
            new_parameters = parameters.copy()
            new_parameters.model_weights = weights.tolist() + [sent_len_weight]
            quality = self._calculate_bleu(new_parameters, tune_source_corpus, tune_target_corpus)
            if eval_count != -1:
                current_step = min(eval_count + 1, self.max_progress_function_evaluations)
                progress(ProgressStatus.from_step(current_step, self.max_progress_function_evaluations))
            return quality

        progress(ProgressStatus.from_step(0, self.max_progress_function_evaluations))
        simplex = NelderMeadSimplex(self.convergence_tolerance, self.max_function_evaluations, 1.0)
        result = simplex.find_minimum(evaluate, parameters.model_weights[:7])

        stats.metrics["bleu"] = 1.0 - result.error_value

        best_parameters = parameters.copy()
        best_parameters.model_weights = result.minimizing_point.tolist() + [sent_len_weight]

        if result.function_evaluation_count < self.max_progress_function_evaluations:
            progress(ProgressStatus(result.function_evaluation_count, 1.0))
        return best_parameters

    def _calculate_bleu(
        self,
        parameters: ThotSmtParameters,
        source_corpus: Sequence[Sequence[str]],
        tune_target_corpus: Sequence[Sequence[str]],
    ) -> float:
        translations = self._generate_translations(parameters, source_corpus)
        bleu = compute_bleu(translations, tune_target_corpus)
        penalty = 0
        for i in range(len(parameters.model_weights)):
            if i == 0 or i == 2 or i == 7:
                continue

            if parameters.model_weights[i] < 0:
                penalty += parameters.model_weights[i] * 1000 * -1
        return (1.0 - bleu) + penalty

    def _generate_translations(
        self, parameters: ThotSmtParameters, source_corpus: Sequence[Sequence[str]]
    ) -> Sequence[Sequence[str]]:
        model: Optional[tt.SmtModel] = None
        decoder: Optional[tt.SmtDecoder] = None
        try:
            model = load_smt_model(self._word_alignment_model_type, parameters)
            decoder = load_smt_decoder(model, parameters)
            translations = decoder.translate_batch([" ".join(escape_tokens(s)) for s in source_corpus])
            return [t.target for t in translations]
        finally:
            if decoder is not None:
                decoder.clear()
            if model is not None:
                model.clear()

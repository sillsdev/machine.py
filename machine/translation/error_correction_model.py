from math import log
from typing import Sequence

from .ecm_score_info import EcmScoreInfo
from .edit_operation import EditOperation
from .segment_edit_distance import SegmentEditDistance
from .translation_result_builder import TranslationResultBuilder
from .translation_sources import TranslationSources


class ErrorCorrectionModel:
    def __init__(self) -> None:
        self._segment_edit_distance = SegmentEditDistance()
        self.set_error_model_parameters(voc_size=128, hit_prob=0.8, ins_factor=1, subst_factor=1, del_factor=1)

    def set_error_model_parameters(
        self, voc_size: int, hit_prob: float, ins_factor: float, subst_factor: float, del_factor: float
    ) -> None:
        if voc_size == 0:
            e = (1 - hit_prob) / (ins_factor + subst_factor + del_factor)
        else:
            e = (1 - hit_prob) / ((ins_factor * voc_size) + (subst_factor * (voc_size - 1)) + del_factor)

        ins_prob = e * ins_factor
        subst_prob = e * subst_factor
        del_prob = e * del_factor

        self._segment_edit_distance.hit_cost = -log(hit_prob)
        self._segment_edit_distance.insertion_cost = -log(ins_prob)
        self._segment_edit_distance.substitution_cost = -log(subst_prob)
        self._segment_edit_distance.deletion_cost = -log(del_prob)

    def setup_initial_esi(self, initial_esi: EcmScoreInfo) -> None:
        score = self._segment_edit_distance.compute([], [])
        initial_esi.scores.clear()
        initial_esi.scores.append(score)
        initial_esi.operations.clear()

    def setup_esi(self, esi: EcmScoreInfo, prev_esi: EcmScoreInfo, word: str) -> None:
        score = self._segment_edit_distance.compute([word], [])
        esi.scores.clear()
        esi.scores.append(prev_esi.scores[0] + score)
        esi.operations.clear()
        esi.operations.append(EditOperation.NONE)

    def extend_initial_esi(
        self, initial_esi: EcmScoreInfo, prev_initial_esi: EcmScoreInfo, prefix_diff: Sequence[str]
    ) -> None:
        self._segment_edit_distance.incr_compute_prefix_first_row(
            initial_esi.scores, prev_initial_esi.scores, prefix_diff
        )

    def extend_esi(
        self,
        esi: EcmScoreInfo,
        prev_esi: EcmScoreInfo,
        word: str,
        prefix_diff: Sequence[str],
        is_last_word_complete: bool,
    ) -> None:
        ops = self._segment_edit_distance.incr_compute_prefix(
            esi.scores, prev_esi.scores, word, prefix_diff, is_last_word_complete
        )
        esi.operations.extend(ops)

    def correct_prefix(
        self,
        builder: TranslationResultBuilder,
        uncorrected_prefix_len: int,
        prefix: Sequence[str],
        is_last_word_complete: bool,
    ) -> int:
        if uncorrected_prefix_len == 0:
            for w in prefix:
                builder.append_token(w, TranslationSources.PREFIX, -1)
            return len(prefix)

        _, word_ops, char_ops = self._segment_edit_distance.compute_prefix(
            builder.target_tokens[:uncorrected_prefix_len],
            prefix,
            is_last_word_complete,
            use_prefix_del_op=False,
        )
        return builder.correct_prefix(word_ops, char_ops, prefix, is_last_word_complete)

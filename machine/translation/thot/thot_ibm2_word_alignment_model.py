from math import exp
from typing import Collection, Sequence, cast

import thot.alignment as ta

from ...corpora.aligned_word_pair import AlignedWordPair
from ..ibm2_word_alignment_model import Ibm2WordAlignmentModel
from .thot_ibm1_word_alignment_model import ThotIbm1WordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotIbm2WordAlignmentModel(ThotIbm1WordAlignmentModel, Ibm2WordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.IBM2

    @property
    def thot_model(self) -> ta.Ibm2AlignmentModel:
        return cast(ta.Ibm2AlignmentModel, self._model)

    def get_alignment_probability(
        self, source_length: int, source_index: int, target_length: int, target_index: int
    ) -> float:
        return exp(self.get_alignment_log_probability(source_length, source_index, target_length, target_index))

    def get_alignment_log_probability(
        self, source_length: int, source_index: int, target_length: int, target_index: int
    ) -> float:
        if target_index == -1:
            return 0
        # add 1 to convert the specified indices to Thot position indices, which are 1-based
        return self.thot_model.alignment_log_prob(target_index + 1, source_length, target_length, source_index + 1)

    def compute_aligned_word_pair_scores(
        self, source_segment: Sequence[str], target_segment: Sequence[str], word_pairs: Collection[AlignedWordPair]
    ) -> None:
        for word_pair in word_pairs:
            if word_pair.target_index == -1:
                word_pair.translation_score = 0
                word_pair.alignment_score = 0
            else:
                source_word = None if word_pair.source_index == -1 else source_segment[word_pair.source_index]
                target_word = target_segment[word_pair.target_index]
                word_pair.translation_score = self.get_translation_probability(source_word, target_word)
                word_pair.alignment_score = self.get_alignment_probability(
                    len(source_segment), word_pair.source_index, len(target_segment), word_pair.target_index
                )

from math import log
from typing import Collection, Sequence, cast

import thot.alignment as ta

from ...corpora.aligned_word_pair import AlignedWordPair
from .thot_word_alignment_model import ThotWordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotIbm1WordAlignmentModel(ThotWordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.IBM1

    @property
    def thot_model(self) -> ta.Ibm1AlignmentModel:
        return cast(ta.Ibm1AlignmentModel, self._model)

    def get_alignment_probability(self, source_length: int) -> float:
        return 1.0 / (source_length + 1)

    def get_alignment_log_probability(self, source_length: int) -> float:
        return log(self.get_alignment_probability(source_length))

    def compute_aligned_word_pair_scores(
        self, source_segment: Sequence[str], target_segment: Sequence[str], word_pairs: Collection[AlignedWordPair]
    ) -> None:
        alignment_score = self.get_alignment_probability(len(source_segment))
        for word_pair in word_pairs:
            if word_pair.target_index == -1:
                word_pair.translation_score = 0
                word_pair.alignment_score = 0
            else:
                source_word = None if word_pair.source_index == -1 else source_segment[word_pair.source_index]
                target_word = target_segment[word_pair.target_index]
                word_pair.translation_score = self.get_translation_probability(source_word, target_word)
                word_pair.alignment_score = alignment_score

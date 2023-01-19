from __future__ import annotations

from math import exp
from typing import Collection, Sequence, Set, cast

import thot.alignment as ta

from ...corpora.aligned_word_pair import AlignedWordPair
from ..hmm_word_alignment_model import HmmWordAlignmentModel
from .thot_ibm1_word_alignment_model import ThotIbm1WordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


class ThotHmmWordAlignmentModel(ThotIbm1WordAlignmentModel, HmmWordAlignmentModel):
    @property
    def type(self) -> ThotWordAlignmentModelType:
        return ThotWordAlignmentModelType.HMM

    @property
    def thot_model(self) -> ta.HmmAlignmentModel:
        return cast(ta.HmmAlignmentModel, self._model)

    def get_alignment_probability(self, source_length: int, prev_source_index: int, source_index: int) -> float:
        return exp(self.get_alignment_log_probability(source_length, prev_source_index, source_index))

    def get_alignment_log_probability(self, source_length: int, prev_source_index: int, source_index: int) -> float:
        # add 1 to convert the specified indices to Thot position indices, which are 1-based
        return self.thot_model.hmm_alignment_log_prob(prev_source_index + 1, source_length, source_index + 1)

    def compute_aligned_word_pair_scores(
        self, source_segment: Sequence[str], target_segment: Sequence[str], word_pairs: Collection[AlignedWordPair]
    ) -> None:
        source_indices = [-2] * len(target_segment)
        prev_source_index = -1
        aligned_target_indices: Set[int] = set()
        for word_pair in sorted(word_pairs, key=lambda wp: wp.target_index):
            if word_pair.target_index == -1:
                continue
            if source_indices[word_pair.target_index] == -2:
                source_index = word_pair.source_index
                if source_index == -1:
                    source_index = -1 if prev_source_index == -1 else len(source_segment) + prev_source_index
                source_indices[word_pair.target_index] = source_index
                prev_source_index = source_index
            aligned_target_indices.add(word_pair.target_index)

        if len(aligned_target_indices) < len(target_segment):
            # there are target words that are aligned to NULL, so fill in the correct source index
            prev_source_index = -1
            for j in range(len(target_segment)):
                if source_indices[j] == -2:
                    source_index = -1 if prev_source_index == -1 else len(source_segment) + prev_source_index
                    source_indices[j] = source_index
                    prev_source_index = source_index
                else:
                    prev_source_index = source_indices[j]

        for word_pair in word_pairs:
            if word_pair.target_index == -1:
                word_pair.translation_score = 0
                word_pair.alignment_score = 0
            else:
                source_word = None if word_pair.source_index == -1 else source_segment[word_pair.source_index]
                target_word = target_segment[word_pair.target_index]
                word_pair.translation_score = self.get_translation_probability(source_word, target_word)
                prev_source_index = -1 if word_pair.target_index == 0 else source_indices[word_pair.target_index - 1]
                source_index = source_indices[word_pair.target_index]
                word_pair.alignment_score = self.get_alignment_probability(
                    len(source_segment), prev_source_index, source_index
                )

    def __enter__(self) -> ThotHmmWordAlignmentModel:
        return self

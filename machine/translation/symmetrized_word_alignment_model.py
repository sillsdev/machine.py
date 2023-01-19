from __future__ import annotations

from typing import Collection, Iterable, Optional, Sequence, Tuple, Union

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from .symmetrized_word_aligner import SymmetrizedWordAligner
from .symmetrized_word_alignment_model_trainer import SymmetrizedWordAlignmentModelTrainer
from .trainer import Trainer
from .word_alignment_model import WordAlignmentModel


class SymmetrizedWordAlignmentModel(SymmetrizedWordAligner, WordAlignmentModel):
    def __init__(
        self, direct_word_alignment_model: WordAlignmentModel, inverse_word_alignment_model: WordAlignmentModel
    ) -> None:
        super().__init__(direct_word_alignment_model, inverse_word_alignment_model)
        self._direct_word_alignment_model = direct_word_alignment_model
        self._inverse_word_alignment_model = inverse_word_alignment_model

    @property
    def direct_word_alignment_model(self) -> WordAlignmentModel:
        return self._direct_word_alignment_model

    @property
    def inverse_word_alignment_model(self) -> WordAlignmentModel:
        return self._inverse_word_alignment_model

    @property
    def source_words(self) -> Sequence[str]:
        return self._direct_word_alignment_model.source_words

    @property
    def target_words(self) -> Sequence[str]:
        return self._direct_word_alignment_model.target_words

    @property
    def special_symbol_indices(self) -> Collection[int]:
        return self._direct_word_alignment_model.special_symbol_indices

    def get_translations(
        self, source_word: Optional[Union[str, int]], threshold: float = 0
    ) -> Iterable[Tuple[int, float]]:
        for target_word_index, dir_score in self._direct_word_alignment_model.get_translations(source_word):
            inv_score = self._inverse_word_alignment_model.get_translation_score(target_word_index, source_word)
            score = max(dir_score, inv_score)
            if score > threshold:
                yield (target_word_index, score)

    def get_translation_score(
        self, source_word: Optional[Union[str, int]], target_word: Optional[Union[str, int]]
    ) -> float:
        dir_score = self._direct_word_alignment_model.get_translation_score(source_word, target_word)
        inv_score = self._inverse_word_alignment_model.get_translation_score(target_word, source_word)
        return max(dir_score, inv_score)

    def create_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        direct_trainer = self._direct_word_alignment_model.create_trainer(corpus)
        inverse_trainer = self._inverse_word_alignment_model.create_trainer(corpus.invert())

        return SymmetrizedWordAlignmentModelTrainer(direct_trainer, inverse_trainer)

    def compute_aligned_word_pair_scores(
        self, source_segment: Sequence[str], target_segment: Sequence[str], word_pairs: Collection[AlignedWordPair]
    ) -> None:
        inverse_word_pairs = [wp.invert() for wp in word_pairs]
        self.direct_word_alignment_model.compute_aligned_word_pair_scores(source_segment, target_segment, word_pairs)
        self.inverse_word_alignment_model.compute_aligned_word_pair_scores(
            target_segment, source_segment, inverse_word_pairs
        )
        for word_pair, inverse_word_pair in zip(word_pairs, inverse_word_pairs):
            word_pair.translation_score = max(word_pair.translation_score, inverse_word_pair.translation_score)
            word_pair.alignment_score = max(word_pair.alignment_score, inverse_word_pair.alignment_score)

    def close(self) -> None:
        self._direct_word_alignment_model.close()
        self._inverse_word_alignment_model.close()

    def __enter__(self) -> SymmetrizedWordAlignmentModel:
        return self

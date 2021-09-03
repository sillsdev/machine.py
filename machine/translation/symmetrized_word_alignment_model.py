import sys
from typing import Collection, Iterable, Optional, Sequence, Tuple, Union

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.token_processors import NO_OP, TokenProcessor
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

    def get_alignment_score(
        self,
        source_length: int,
        prev_source_index: int,
        source_index: int,
        target_length: int,
        prev_target_index: int,
        target_index: int,
    ) -> float:
        dir_score = self._direct_word_alignment_model.get_alignment_score(
            source_length, prev_source_index, source_index, target_length, prev_target_index, target_index
        )
        inv_score = self._inverse_word_alignment_model.get_alignment_score(
            target_length, prev_target_index, target_index, source_length, prev_source_index, source_index
        )
        return max(dir_score, inv_score)

    def create_trainer(
        self,
        corpus: ParallelTextCorpus,
        source_preprocessor: TokenProcessor = NO_OP,
        target_preprocessor: TokenProcessor = NO_OP,
        max_corpus_count: int = sys.maxsize,
    ) -> Trainer:
        direct_trainer = self._direct_word_alignment_model.create_trainer(
            corpus, source_preprocessor, target_preprocessor, max_corpus_count
        )
        inverse_trainer = self._inverse_word_alignment_model.create_trainer(
            corpus.invert(), target_preprocessor, source_preprocessor, max_corpus_count
        )

        return SymmetrizedWordAlignmentModelTrainer(direct_trainer, inverse_trainer)

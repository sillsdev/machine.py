from pathlib import Path

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...tokenization.tokenizer import Tokenizer
from ...translation.symmetrized_word_alignment_model_trainer import SymmetrizedWordAlignmentModelTrainer
from ...translation.thot.thot_symmetrized_word_alignment_model import ThotSymmetrizedWordAlignmentModel
from ...translation.thot.thot_word_alignment_model_trainer import ThotWordAlignmentModelTrainer
from ...translation.thot.thot_word_alignment_model_utils import create_thot_word_alignment_model
from ...translation.trainer import Trainer
from ...translation.word_alignment_model import WordAlignmentModel
from ..word_alignment_model_factory import WordAlignmentModelFactory


class ThotWordAlignmentModelFactory(WordAlignmentModelFactory):

    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer:
        (self._model_dir / "tm").mkdir(parents=True, exist_ok=True)
        direct_trainer = ThotWordAlignmentModelTrainer(
            self._config.thot_align.model_type,
            corpus.lowercase(),
            prefix_filename=self._direct_model_path,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
        )
        inverse_trainer = ThotWordAlignmentModelTrainer(
            self._config.thot_align.model_type,
            corpus.invert().lowercase(),
            prefix_filename=self._inverse_model_path,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
        )
        return SymmetrizedWordAlignmentModelTrainer(direct_trainer, inverse_trainer)

    def create_alignment_model(
        self,
    ) -> WordAlignmentModel:
        model = ThotSymmetrizedWordAlignmentModel(
            create_thot_word_alignment_model(self._config.thot_align.model_type, self._direct_model_path),
            create_thot_word_alignment_model(self._config.thot_align.model_type, self._inverse_model_path),
        )
        model.heuristic = self._config.thot_align.word_alignment_heuristic
        return model

    @property
    def _direct_model_path(self) -> Path:
        return self._model_dir / "tm" / "src_trg_invswm"

    @property
    def _inverse_model_path(self) -> Path:
        return self._model_dir / "tm" / "src_trg_swm"

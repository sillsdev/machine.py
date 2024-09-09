import shutil
from typing import Optional

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.text_corpus import TextCorpus
from ...tokenization.detokenizer import Detokenizer
from ...tokenization.tokenizer import Tokenizer
from ...translation.thot.thot_smt_model import ThotSmtModel
from ...translation.thot.thot_smt_model_trainer import ThotSmtModelTrainer
from ...translation.trainer import Trainer
from ...translation.translation_engine import TranslationEngine
from ...translation.truecaser import Truecaser
from ...translation.unigram_truecaser import UnigramTruecaser, UnigramTruecaserTrainer
from ..smt_model_factory import SmtModelFactory
from . import _THOT_NEW_MODEL_DIRECTORY


class ThotSmtModelFactory(SmtModelFactory):
    def init(self) -> None:
        shutil.copytree(_THOT_NEW_MODEL_DIRECTORY, self._model_dir, dirs_exist_ok=True)

    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer:
        return ThotSmtModelTrainer(
            word_alignment_model_type=self._config.thot_mt.word_alignment_model_type,
            corpus=corpus,
            config=self._model_dir / "smt.cfg",
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            lowercase_source=True,
            lowercase_target=True,
        )

    def create_engine(
        self,
        tokenizer: Tokenizer[str, int, str],
        detokenizer: Detokenizer[str, str],
        truecaser: Optional[Truecaser] = None,
    ) -> TranslationEngine:
        return ThotSmtModel(
            word_alignment_model_type=self._config.thot_mt.word_alignment_model_type,
            config=self._model_dir / "smt.cfg",
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            target_detokenizer=detokenizer,
            lowercase_source=True,
            lowercase_target=True,
            truecaser=truecaser,
        )

    def create_truecaser_trainer(self, tokenizer: Tokenizer[str, int, str], target_corpus: TextCorpus) -> Trainer:
        return UnigramTruecaserTrainer(
            corpus=target_corpus, model_path=self._model_dir / "unigram-casing-model.txt", tokenizer=tokenizer
        )

    def create_truecaser(self) -> Truecaser:
        return UnigramTruecaser(model_path=self._model_dir / "unigram-casing-model.txt")

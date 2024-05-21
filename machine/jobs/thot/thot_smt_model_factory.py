import os
import shutil
from pathlib import Path
from typing import Any

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.text_corpus import TextCorpus
from ...tokenization.detokenizer import Detokenizer
from ...tokenization.latin_word_detokenizer import LatinWordDetokenizer
from ...tokenization.latin_word_tokenizer import LatinWordTokenizer
from ...tokenization.tokenizer import Tokenizer
from ...tokenization.whitespace_detokenizer import WhitespaceDetokenizer
from ...tokenization.whitespace_tokenizer import WhitespaceTokenizer
from ...tokenization.zwsp_word_detokenizer import ZwspWordDetokenizer
from ...tokenization.zwsp_word_tokenizer import ZwspWordTokenizer
from ...translation.thot.thot_smt_model import ThotSmtModel
from ...translation.thot.thot_smt_model_trainer import ThotSmtModelTrainer
from ...translation.trainer import Trainer
from ...translation.translation_engine import TranslationEngine
from ...translation.truecaser import Truecaser
from ...translation.unigram_truecaser import UnigramTruecaser, UnigramTruecaserTrainer
from ..smt_model_factory import SmtModelFactory

_THOT_NEW_MODEL_DIRECTORY = os.path.join(os.path.dirname(__file__), "thot-new-model")

_TOKENIZERS = ["latin", "whitespace", "zwsp"]


class ThotSmtModelFactory(SmtModelFactory):
    def __init__(self, config: Any) -> None:
        self._config = config

    def init(self) -> None:
        shutil.copytree(_THOT_NEW_MODEL_DIRECTORY, self._model_dir, dirs_exist_ok=True)

    def create_tokenizer(self) -> Tokenizer[str, int, str]:
        name: str = self._config.thot.tokenizer
        name = name.lower()
        if name == "latin":
            return LatinWordTokenizer()
        if name == "whitespace":
            return WhitespaceTokenizer()
        if name == "zwsp":
            return ZwspWordTokenizer()
        raise RuntimeError(f"Unknown tokenizer: {name}.  Available tokenizers are: {_TOKENIZERS}.")

    def create_detokenizer(self) -> Detokenizer[str, str]:
        name: str = self._config.thot.tokenizer
        name = name.lower()
        if name == "latin":
            return LatinWordDetokenizer()
        if name == "whitespace":
            return WhitespaceDetokenizer()
        if name == "zwsp":
            return ZwspWordDetokenizer()
        raise RuntimeError(f"Unknown detokenizer: {name}.  Available detokenizers are: {_TOKENIZERS}.")

    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer:
        return ThotSmtModelTrainer(
            word_alignment_model_type=self._config.thot.word_alignment_model_type,
            corpus=corpus,
            config=self._model_dir / "smt.cfg",
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            lowercase_source=True,
            lowercase_target=True,
        )

    def create_engine(
        self, tokenizer: Tokenizer[str, int, str], detokenizer: Detokenizer[str, str], truecaser: Truecaser
    ) -> TranslationEngine:
        return ThotSmtModel(
            word_alignment_model_type=self._config.thot.word_alignment_model_type,
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

    def save_model(self) -> Path:
        tar_file_basename = Path(
            self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model"
        )
        return Path(shutil.make_archive(str(tar_file_basename), "zip", self._model_dir))

    @property
    def _model_dir(self) -> Path:
        return Path(self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model")

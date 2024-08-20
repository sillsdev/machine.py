import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.translation_engine import TranslationEngine
from ..translation.truecaser import Truecaser


class SmtModelFactory(ABC):
    def __init__(self, config: Any) -> None:
        self._config = config

    def init(self) -> None:
        pass

    @abstractmethod
    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer: ...

    @abstractmethod
    def create_engine(
        self,
        tokenizer: Tokenizer[str, int, str],
        detokenizer: Detokenizer[str, str],
        truecaser: Optional[Truecaser] = None,
    ) -> TranslationEngine: ...

    @abstractmethod
    def create_truecaser_trainer(self, tokenizer: Tokenizer[str, int, str], target_corpus: TextCorpus) -> Trainer: ...

    @abstractmethod
    def create_truecaser(self) -> Truecaser: ...

    def save_model(self) -> Path:
        tar_file_basename = Path(
            self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model"
        )
        return Path(shutil.make_archive(str(tar_file_basename), "gztar", self._model_dir))

    @property
    def _model_dir(self) -> Path:
        return Path(self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model")

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.word_alignment_model import WordAlignmentModel


class WordAlignmentModelFactory(ABC):
    def __init__(self, config: Any) -> None:
        self._config = config

    def init(self) -> None:
        pass

    @abstractmethod
    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer: ...

    @abstractmethod
    def create_alignment_model(
        self,
    ) -> WordAlignmentModel: ...

    def save_model(self) -> Path:
        tar_file_basename = Path(
            self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model"
        )
        return Path(shutil.make_archive(str(tar_file_basename), "gztar", self._model_dir))

    @property
    def _model_dir(self) -> Path:
        return Path(self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model")

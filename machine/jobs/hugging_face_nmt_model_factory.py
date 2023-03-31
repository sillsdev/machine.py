from pathlib import Path
from typing import Any, Optional

from transformers import Seq2SeqTrainingArguments

from machine.corpora.parallel_text_corpus import ParallelTextCorpus
from machine.corpora.text_corpus import TextCorpus
from machine.translation.trainer import Trainer

from ..translation.huggingface.hugging_face_nmt_model import HuggingFaceNmtModel
from ..translation.huggingface.hugging_face_nmt_model_trainer import HuggingFaceNmtModelTrainer
from ..translation.translation_model import TranslationModel
from .nmt_model_factory import NmtModelFactory
from .shared_file_service import SharedFileService


class HuggingFaceNmtModelFactory(NmtModelFactory):
    def __init__(self, config: Any, shared_file_service: SharedFileService) -> None:
        self._config = config
        self._shared_file_service = shared_file_service

    def init(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def create_model(self) -> TranslationModel:
        return HuggingFaceNmtModel(
            self._model_dir,
            self._config.parent_model_name,
            source_lang=self._config.src_lang,
            target_lang=self._config.trg_lang,
            device=0,
        )

    def create_source_tokenizer_trainer(self, corpus: TextCorpus) -> Optional[Trainer]:
        return None

    def create_target_tokenizer_trainer(self, corpus: TextCorpus) -> Optional[Trainer]:
        return None

    def create_model_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        return HuggingFaceNmtModelTrainer(
            self._config.parent_model_name,
            Seq2SeqTrainingArguments(output_dir=str(self._model_dir), overwrite_output_dir=True),
            corpus,
            self._config.src_lang,
            self._config.trg_lang,
        )

    def save_model(self) -> None:
        self._shared_file_service.save_model(self._model_dir)

    @property
    def _model_dir(self) -> Path:
        return Path(self._config.data_dir, "builds", self._config.build_id, "model")

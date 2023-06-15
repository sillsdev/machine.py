from pathlib import Path
from typing import Any, cast

from transformers import AutoConfig, AutoModelForSeq2SeqLM, HfArgumentParser, PreTrainedModel, Seq2SeqTrainingArguments

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.text_corpus import TextCorpus
from ...translation.huggingface.hugging_face_nmt_engine import HuggingFaceNmtEngine
from ...translation.huggingface.hugging_face_nmt_model_trainer import HuggingFaceNmtModelTrainer
from ...translation.null_trainer import NullTrainer
from ...translation.trainer import Trainer
from ...translation.translation_engine import TranslationEngine
from ..nmt_model_factory import NmtModelFactory
from ..shared_file_service import SharedFileService


class HuggingFaceNmtModelFactory(NmtModelFactory):
    def __init__(self, config: Any, shared_file_service: SharedFileService) -> None:
        self._config = config
        self._shared_file_service = shared_file_service
        args = config.huggingface.train_params.to_dict()
        args["output_dir"] = str(self._model_dir)
        args["overwrite_output_dir"] = True
        if "max_steps" in self._config:
            args["max_steps"] = self._config.max_steps
        parser = HfArgumentParser(cast(Any, Seq2SeqTrainingArguments))
        self._training_args = cast(Seq2SeqTrainingArguments, parser.parse_dict(args)[0])
        if (
            not config.clearml
            and self._training_args.report_to is not None
            and "clearml" in self._training_args.report_to
        ):
            self._training_args.report_to.remove("clearml")

    @property
    def train_tokenizer(self) -> bool:
        return False

    def init(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        config = AutoConfig.from_pretrained(
            self._config.huggingface.parent_model_name, label2id={}, id2label={}, num_labels=0
        )
        self._model = cast(
            PreTrainedModel,
            AutoModelForSeq2SeqLM.from_pretrained(self._config.huggingface.parent_model_name, config=config),
        )

    def create_source_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        return NullTrainer()

    def create_target_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        return NullTrainer()

    def create_model_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        return HuggingFaceNmtModelTrainer(
            self._model,
            self._training_args,
            corpus,
            src_lang=self._config.src_lang,
            tgt_lang=self._config.trg_lang,
        )

    def create_engine(self) -> TranslationEngine:
        return HuggingFaceNmtEngine(
            self._model,
            src_lang=self._config.src_lang,
            tgt_lang=self._config.trg_lang,
            device=self._config.huggingface.generate_params.device,
            num_beams=self._config.huggingface.generate_params.num_beams,
            batch_size=self._config.huggingface.generate_params.batch_size,
        )

    def save_model(self) -> None:
        self._shared_file_service.save_model(self._model_dir)

    @property
    def _model_dir(self) -> Path:
        return Path(self._config.data_dir, "builds", self._config.build_id, "model")

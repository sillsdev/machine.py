import logging
import tarfile
from pathlib import Path
from typing import Any, cast

from transformers import AutoConfig, AutoModelForSeq2SeqLM, HfArgumentParser, PreTrainedModel, Seq2SeqTrainingArguments
from transformers.integrations import ClearMLCallback
from transformers.tokenization_utils import TruncationStrategy

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.text_corpus import TextCorpus
from ...translation.huggingface.hugging_face_nmt_engine import HuggingFaceNmtEngine
from ...translation.huggingface.hugging_face_nmt_model_trainer import HuggingFaceNmtModelTrainer
from ...translation.null_trainer import NullTrainer
from ...translation.trainer import Trainer
from ...translation.translation_engine import TranslationEngine
from ..nmt_model_factory import NmtModelFactory

logger = logging.getLogger(__name__)


class HuggingFaceNmtModelFactory(NmtModelFactory):
    def __init__(self, config: Any) -> None:
        self._config = config
        args = config.huggingface.train_params.to_dict()
        args["output_dir"] = str(self._model_dir)
        args["overwrite_output_dir"] = True
        # Use "max_steps" from root for backward compatibility
        if "max_steps" in self._config.huggingface:
            args["max_steps"] = self._config.huggingface.max_steps
        parser = HfArgumentParser(cast(Any, Seq2SeqTrainingArguments))
        self._training_args = cast(Seq2SeqTrainingArguments, parser.parse_dict(args)[0])
        if self._training_args.max_steps > 50000:
            raise ValueError("max_steps must be less than or equal to 50000")
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
            add_unk_src_tokens=self._config.huggingface.tokenizer.add_unk_src_tokens,
            add_unk_trg_tokens=self._config.huggingface.tokenizer.add_unk_trg_tokens,
        )

    def create_engine(self) -> TranslationEngine:
        return HuggingFaceNmtEngine(
            self._model,
            src_lang=self._config.src_lang,
            tgt_lang=self._config.trg_lang,
            device=self._config.huggingface.generate_params.device,
            num_beams=self._config.huggingface.generate_params.num_beams,
            batch_size=self._config.huggingface.generate_params.batch_size,
            truncation=TruncationStrategy.LONGEST_FIRST,
            oom_batch_size_backoff_mult=self._config.huggingface.generate_params.oom_batch_size_backoff_mult,
        )

    def save_model(self) -> Path:
        tar_file_path = Path(
            self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model.tar.gz"
        )
        with tarfile.open(tar_file_path, "w:gz") as tar:
            for path in self._model_dir.iterdir():
                if path.is_file():
                    tar.add(path, arcname=path.name)
        return tar_file_path

    @property
    def _model_dir(self) -> Path:
        return Path(self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model")


# FIXME - remove this code when the fix is applied to Huggingface
# https://github.com/huggingface/transformers/pull/26763
def on_train_end(
    self: ClearMLCallback, args, state, control, model=None, tokenizer=None, metrics=None, logs=None, **kwargs
):
    pass


setattr(ClearMLCallback, "on_train_end", on_train_end)

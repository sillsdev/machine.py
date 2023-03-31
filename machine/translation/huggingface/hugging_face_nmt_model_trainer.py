import logging
import os
import sys
from typing import Any, Callable, Optional, Union, cast

import datasets.utils.logging as datasets_logging
import transformers.utils.logging as transformers_logging
from datasets.arrow_dataset import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    M2M100Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.models.mbart50 import MBart50Tokenizer
from transformers.trainer_utils import get_last_checkpoint

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...utils.progress_status import ProgressStatus
from ..trainer import Trainer, TrainStats

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

MULTILINGUAL_TOKENIZERS = (
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer,
    NllbTokenizer,
    NllbTokenizerFast,
)


class HuggingFaceNmtModelTrainer(Trainer):
    def __init__(
        self,
        parent_model_name: str,
        training_args: Seq2SeqTrainingArguments,
        corpus: Union[ParallelTextCorpus, Dataset],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> None:
        self._parent_model_name = parent_model_name
        self._training_args = training_args
        self._corpus = corpus
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._trainer: Optional[Seq2SeqTrainer] = None
        self._metrics = {}
        self.max_source_length = 200
        self.max_target_length = 200

    @property
    def stats(self) -> TrainStats:
        return super().stats

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        if self._training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers_logging.set_verbosity_info()

        log_level = self._training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets_logging.set_verbosity(log_level)
        transformers_logging.set_verbosity(log_level)
        transformers_logging.enable_default_handler()
        transformers_logging.enable_explicit_format()

        last_checkpoint = None
        if os.path.isdir(self._training_args.output_dir) and not self._training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self._training_args.output_dir)
            if last_checkpoint is None and any(os.path.isfile(p) for p in os.listdir(self._training_args.output_dir)):
                raise ValueError(
                    f"Output directory ({self._training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and self._training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Set seed before initializing model.
        set_seed(self._training_args.seed)

        config = AutoConfig.from_pretrained(
            self._parent_model_name, use_cache=not self._training_args.gradient_checkpointing
        )
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(self._parent_model_name, config=config)
        tokenizer: Any = AutoTokenizer.from_pretrained(self._parent_model_name, use_fast=True)

        def add_lang_code_to_tokenizer(tokenizer: Any, lang_code: str):
            if lang_code in tokenizer.lang_code_to_id:
                return
            tokenizer.add_special_tokens(
                {"additional_special_tokens": tokenizer.additional_special_tokens + [lang_code]}
            )
            lang_id = tokenizer.convert_tokens_to_ids(lang_code)
            tokenizer.lang_code_to_id[lang_code] = lang_id
            if isinstance(tokenizer, (NllbTokenizer, MBart50Tokenizer, MBartTokenizer)):
                tokenizer.id_to_lang_code[lang_id] = lang_code
                tokenizer.fairseq_tokens_to_ids[lang_code] = lang_id
                tokenizer.fairseq_ids_to_tokens[lang_id] = lang_code
            elif isinstance(tokenizer, M2M100Tokenizer):
                tokenizer.lang_token_to_id[lang_code] = lang_id
                tokenizer.id_to_lang_token[lang_id] = lang_code

        if isinstance(tokenizer, MULTILINGUAL_TOKENIZERS):
            if self._source_lang is not None:
                add_lang_code_to_tokenizer(tokenizer, self._source_lang)
            if self._target_lang is not None:
                add_lang_code_to_tokenizer(tokenizer, self._target_lang)

        # We resize the embeddings only when necessary to avoid index errors.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        # Set decoder_start_token_id
        if (
            self._target_lang is not None
            and model.config.decoder_start_token_id is None
            and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast))
        ):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[self._target_lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(self._target_lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        # For translation we set the codes of our source and target languages (only useful for mBART, the others will
        # ignore those attributes).
        if isinstance(tokenizer, MULTILINGUAL_TOKENIZERS):
            if self._source_lang is None or self._target_lang is None:
                raise ValueError(
                    f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires source_lang and "
                    "target_lang to be set."
                )

            tokenizer.src_lang = self._source_lang
            tokenizer.tgt_lang = self._target_lang

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
            forced_bos_token_id = tokenizer.lang_code_to_id[self._target_lang]
            model.config.forced_bos_token_id = forced_bos_token_id

        prefix = ""
        if self._parent_model_name.startswith("t5-") or self._parent_model_name.startswith("mt5-"):
            prefix = f"translate {self._source_lang} to {self._target_lang}: "

        source_lang = self._source_lang
        if source_lang is None:
            source_lang = "src"
        target_lang = self._target_lang
        if target_lang is None:
            target_lang = "trg"

        def preprocess_function(examples):
            inputs = [ex[source_lang] for ex in examples["translation"]]
            targets = [ex[target_lang] for ex in examples["translation"]]
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=self.max_source_length, truncation=True)

            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=targets, max_length=self.max_target_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        if isinstance(self._corpus, Dataset):
            train_count = len(self._corpus)
            train_dataset = self._corpus
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on train dataset",
            )
        else:
            train_count = self._corpus.count(include_empty=False)

            train_dataset = self._corpus.filter_nonempty().to_hf_dataset(source_lang, target_lang)
            train_dataset = train_dataset.map(
                preprocess_function, batched=True, remove_columns=["text", "ref", "translation", "alignment"]
            )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if self._training_args.fp16 else None,
        )

        self._trainer = Seq2SeqTrainer(
            model=model,
            args=self._training_args,
            train_dataset=cast(Any, train_dataset),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        checkpoint = None
        if self._training_args.resume_from_checkpoint is not None:
            checkpoint = self._training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = self._trainer.train(resume_from_checkpoint=checkpoint)

        self._metrics = train_result.metrics
        self._metrics["train_samples"] = train_count

        self._trainer.log_metrics("train", self._metrics)

    def save(self) -> None:
        if self._trainer is None:
            raise RuntimeError("The model has not been trained.")
        self._trainer.save_model()
        self._trainer.save_metrics("train", self._metrics)
        self._trainer.save_state()

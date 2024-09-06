import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, cast

import datasets.utils.logging as datasets_logging
import torch  # pyright: ignore[reportMissingImports]
import transformers.utils.logging as transformers_logging
from datasets.arrow_dataset import Dataset
from sacremoses import MosesPunctNormalizer
from torch import Tensor  # pyright: ignore[reportMissingImports]
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportMissingImports] # noqa: F401
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.models.mbart50 import MBart50Tokenizer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...utils.progress_status import ProgressStatus
from ..trainer import Trainer, TrainStats

logger = logging.getLogger(__name__)


def prepare_decoder_input_ids_from_labels(self: M2M100ForConditionalGeneration, labels: Tensor) -> Tensor:
    # shift ids to the right
    shifted_input_ids = labels.new_zeros(labels.shape)
    shifted_input_ids[:, 1:] = labels[:, :-1].clone()
    assert self.config.decoder_start_token_id is not None
    shifted_input_ids[:, 0] = self.config.decoder_start_token_id

    if self.config.pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.config.pad_token_id)

    return shifted_input_ids


setattr(
    M2M100ForConditionalGeneration,
    "prepare_decoder_input_ids_from_labels",
    prepare_decoder_input_ids_from_labels,
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
        model: Union[PreTrainedModel, str],
        training_args: Seq2SeqTrainingArguments,
        corpus: Union[ParallelTextCorpus, Dataset],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        add_unk_src_tokens: bool = False,
        add_unk_trg_tokens: bool = True,
    ) -> None:
        self._model = model
        self._training_args = training_args
        self._corpus = corpus
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._trainer: Optional[Seq2SeqTrainer] = None
        self._metrics = {}
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self._add_unk_src_tokens = add_unk_src_tokens
        self._add_unk_trg_tokens = add_unk_trg_tokens
        self._mpn = MosesPunctNormalizer()
        self._mpn.substitutions = [(re.compile(r), sub) for r, sub in self._mpn.substitutions]
        self._stats = TrainStats()

    @property
    def stats(self) -> TrainStats:
        return self._stats

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

        if isinstance(self._model, PreTrainedModel):
            model = self._model
            self._original_use_cache = model.config.use_cache
            model.config.use_cache = not self._training_args.gradient_checkpointing
        else:
            config = AutoConfig.from_pretrained(
                self._model,
                use_cache=not self._training_args.gradient_checkpointing,
                label2id={},
                id2label={},
                num_labels=0,
            )
            model = cast(PreTrainedModel, AutoModelForSeq2SeqLM.from_pretrained(self._model, config=config))

        logger.info("Initializing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, use_fast=True)

        src_lang = self._src_lang
        if src_lang is None:
            src_lang = "src"
        tgt_lang = self._tgt_lang
        if tgt_lang is None:
            tgt_lang = "tgt"

        if isinstance(self._corpus, Dataset):
            train_dataset = self._corpus
        else:
            train_dataset = self._corpus.filter_nonempty().to_hf_dataset(src_lang, tgt_lang)

        def find_missing_characters(tokenizer: Any, train_dataset: Dataset, lang_codes: List[str]) -> List[str]:
            vocab = tokenizer.get_vocab().keys()
            charset = set()
            mpn_normalize = True if isinstance(tokenizer, (NllbTokenizerFast)) else False
            for ex in train_dataset["translation"]:
                for lang_code in lang_codes:
                    ex_text = ex[lang_code]
                    if mpn_normalize:
                        ex_text = self._mpn.normalize(ex_text)
                    ex_text = tokenizer.backend_tokenizer.normalizer.normalize_str(ex_text)
                    charset = charset | set(ex_text)
            charset = set(filter(None, {char.strip() for char in charset}))
            missing_characters = sorted(list(charset - vocab))
            return missing_characters

        def add_tokens(tokenizer: Any, missing_tokens: List[str]) -> Any:
            tokenizer_dir = Path(self._training_args.output_dir)
            tokenizer.save_pretrained(str(tokenizer_dir))
            with open(tokenizer_dir / "tokenizer.json", "r+", encoding="utf-8") as file:
                data = json.load(file)
                vocab_len = len(tokenizer)
                if isinstance(data["model"]["vocab"], dict):
                    for i, token in enumerate(missing_tokens):
                        data["model"]["vocab"][token] = vocab_len + i
                elif isinstance(data["model"]["vocab"], list):
                    for i, token in enumerate(missing_tokens):
                        data["model"]["vocab"].append([token, vocab_len + i])
                file.seek(0)
                json.dump(data, file, ensure_ascii=False, indent=4)
                file.truncate()
            logger.info(f"Added {len(missing_tokens)} tokens to the tokenizer: {missing_tokens}")
            return AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)

        if self._add_unk_src_tokens or self._add_unk_trg_tokens:
            logger.info("Checking for missing tokens")
            if not isinstance(tokenizer, PreTrainedTokenizerFast):
                logger.warning(
                    f"Tokenizer can not be updated from default configuration: \
                        tokenizer type {type(tokenizer)} is not an instance of PreTrainedTokenizerFast."
                )
            else:
                norm_tok = PreTrainedTokenizerFast.from_pretrained(
                    str(Path(os.path.dirname(os.path.abspath(__file__))) / "custom_normalizer"),
                    use_fast=True,
                )
                # using unofficially supported behavior to set the normalizer
                tokenizer.backend_tokenizer.normalizer = norm_tok.backend_tokenizer.normalizer  # type: ignore
                if self._add_unk_src_tokens and self._add_unk_trg_tokens:
                    lang_codes = [src_lang, tgt_lang]
                elif self._add_unk_src_tokens:
                    lang_codes = [src_lang]
                else:
                    lang_codes = [tgt_lang]
                missing_tokens = find_missing_characters(tokenizer, train_dataset, lang_codes)
                if missing_tokens:
                    tokenizer = add_tokens(tokenizer, missing_tokens)

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
                tokenizer.lang_code_to_token[lang_code] = lang_code
                tokenizer.lang_token_to_id[lang_code] = lang_id
                tokenizer.id_to_lang_token[lang_id] = lang_code

        if isinstance(tokenizer, MULTILINGUAL_TOKENIZERS):
            logger.info("Add new language codes as tokens")
            if self._src_lang is not None:
                add_lang_code_to_tokenizer(tokenizer, self._src_lang)
            if self._tgt_lang is not None:
                add_lang_code_to_tokenizer(tokenizer, self._tgt_lang)

        # We resize the embeddings only when necessary to avoid index errors.
        embedding_size = cast(Any, model.get_input_embeddings()).weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        # Set decoder_start_token_id
        if (
            self._tgt_lang is not None
            and model.config.decoder_start_token_id is None
            and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast))
        ):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[self._tgt_lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(self._tgt_lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        # For translation we set the codes of our source and target languages (only useful for mBART, the others will
        # ignore those attributes).
        if isinstance(tokenizer, MULTILINGUAL_TOKENIZERS):
            if self._src_lang is None or self._tgt_lang is None:
                raise ValueError(
                    f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires source_lang and "
                    "target_lang to be set."
                )

            tokenizer.src_lang = self._src_lang
            tokenizer.tgt_lang = self._tgt_lang

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
            forced_bos_token_id = tokenizer.lang_code_to_id[self._tgt_lang]
            model.config.forced_bos_token_id = forced_bos_token_id
            if model.generation_config is not None:
                model.generation_config.forced_bos_token_id = forced_bos_token_id

        prefix = ""
        if model.name_or_path.startswith("t5-") or model.name_or_path.startswith("google/mt5-"):
            prefix = f"translate {self._src_lang} to {self._tgt_lang}: "

        max_source_length = self.max_source_length
        if max_source_length is None:
            max_source_length = model.config.max_length
        max_target_length = self.max_target_length
        if max_target_length is None:
            max_target_length = model.config.max_length

        if self._training_args.label_smoothing_factor > 0 and not hasattr(
            model, "prepare_decoder_input_ids_from_labels"
        ):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more "
                "memory"
            )

        def preprocess_function(examples):
            if isinstance(tokenizer, (NllbTokenizer, NllbTokenizerFast)):
                inputs = [self._mpn.normalize(prefix + ex[src_lang]) for ex in examples["translation"]]
                targets = [self._mpn.normalize(ex[tgt_lang]) for ex in examples["translation"]]
            else:
                inputs = [prefix + ex[src_lang] for ex in examples["translation"]]
                targets = [ex[tgt_lang] for ex in examples["translation"]]

            model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        logger.info("Run tokenizer")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
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
            callbacks=[
                _ProgressCallback(
                    self._training_args.max_steps if self._training_args.max_steps > 0 else None,
                    progress,
                    check_canceled,
                )
            ],
        )

        logger.info("Train NMT model")
        ckpt = None
        if self._training_args.resume_from_checkpoint is not None:
            ckpt = self._training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            ckpt = last_checkpoint
        train_result = self._trainer.train(
            resume_from_checkpoint=ckpt,
        )

        self._metrics = train_result.metrics
        self._metrics["train_samples"] = len(train_dataset)
        self._stats.train_corpus_size = self._metrics["train_samples"]

        self._trainer.log_metrics("train", self._metrics)
        logger.info("Model training finished")

    def save(self) -> None:
        if self._trainer is None:
            raise RuntimeError("The model has not been trained.")
        self._trainer.save_model()
        self._trainer.save_metrics("train", self._metrics)
        self._trainer.save_state()
        if isinstance(self._model, PreTrainedModel):
            self._model.name_or_path = self._training_args.output_dir
            self._model.config.name_or_path = self._training_args.output_dir
            self._model.config.use_cache = self._original_use_cache

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        if self._trainer is not None:
            self._trainer = None
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()


class _ProgressCallback(TrainerCallback):
    def __init__(
        self,
        max_steps: Optional[int],
        progress: Optional[Callable[[ProgressStatus], None]],
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        self._max_steps = max_steps
        self._progress = progress
        self._check_canceled = check_canceled

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self._check_canceled is not None:
            self._check_canceled()

        if self._progress is not None and state.is_local_process_zero:
            self._progress(
                ProgressStatus(0) if self._max_steps is None else ProgressStatus.from_step(0, self._max_steps)
            )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self._check_canceled is not None:
            self._check_canceled()

        if self._progress is not None and state.is_local_process_zero:
            self._progress(
                ProgressStatus(state.global_step)
                if self._max_steps is None
                else ProgressStatus.from_step(state.global_step, self._max_steps)
            )

import logging
import os
from pathlib import Path
from typing import Optional, Set

import tensorflow as tf
from opennmt import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN, load_config
from opennmt.data import Vocab

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.sentencepiece.sentence_piece_detokenizer import SentencePieceDetokenizer
from ..tokenization.sentencepiece.sentence_piece_tokenizer import SentencePieceTokenizer
from ..tokenization.sentencepiece.sentence_piece_trainer import SentencePieceTrainer
from ..tokenization.tokenizer import Tokenizer
from ..translation.tensorflow.open_nmt_model import OpenNmtModel
from ..translation.tensorflow.open_nmt_model_trainer import OpenNmtModelTrainer
from ..translation.trainer import Trainer
from ..translation.translation_model import TranslationModel
from .nmt_model_factory import NmtModelFactory


class OpenNmtModelFactory(NmtModelFactory):
    def __init__(self, config: dict) -> None:
        self._config = config

    def init(self) -> None:
        _set_tf_log_level()
        self._model_dir.mkdir(exist_ok=True)

    def create_model(self) -> TranslationModel:
        model_config = self._create_model_config()
        model_type: str = self._config["model"]
        mixed_precision: bool = self._config["mixed_precision"]
        return OpenNmtModel(model_type, model_config, mixed_precision=mixed_precision)

    def create_model_trainer(
        self, source_language_tag: str, target_language_tag: str, corpus: ParallelTextCorpus
    ) -> Trainer:
        model_dir = self._model_dir
        _convert_vocab(model_dir / "src-sp.vocab", model_dir / "src.vocab")
        _convert_vocab(model_dir / "trg-sp.vocab", model_dir / "trg.vocab")

        corpus = corpus.tokenize(self.create_source_tokenizer(), self._create_target_tokenizer())

        # TODO: Add support for parent models
        model_config = self._create_model_config()
        model_type: str = self._config["model"]
        mixed_precision: bool = self._config["mixed_precision"]
        parent_config = self._get_parent_config(source_language_tag, target_language_tag)
        return OpenNmtModelTrainer(
            model_type,
            model_config,
            corpus,
            parent_config,
            mixed_precision,
            resume=True,
            replace_on_save=False,
        )

    def create_source_tokenizer(self) -> Tokenizer[str, int, str]:
        return SentencePieceTokenizer(self._model_dir / "src-sp.model")

    def create_source_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        src_sp_model_prefix = self._model_dir / "src-sp"
        return SentencePieceTrainer(
            corpus,
            vocab_size=8000,
            hard_vocab_limit=False,
            character_coverage=1.0,
            model_prefix=str(src_sp_model_prefix),
            normalization_rule_name="nmt_nfkc_cf",
        )

    def create_target_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        trg_sp_model_prefix = self._model_dir / "trg-sp"
        return SentencePieceTrainer(
            corpus,
            vocab_size=8000,
            hard_vocab_limit=False,
            character_coverage=1.0,
            model_prefix=str(trg_sp_model_prefix),
            normalization_rule_name="nmt_nfkc",
        )

    def create_target_detokenizer(self) -> Detokenizer[str, str]:
        return SentencePieceDetokenizer()

    @property
    def _model_dir(self) -> Path:
        return Path(self._config["models_dir"], self._config["build_id"])

    def _create_target_tokenizer(self) -> Tokenizer[str, int, str]:
        return SentencePieceTokenizer(self._model_dir / "trg-sp.model")

    def _create_model_config(self) -> dict:
        config = {
            "auto_config": True,
            "model_dir": str(self._model_dir),
            "data": {
                "source_vocabulary": "src.vocab",
                "target_vocabulary": "trg.vocab",
                "train_features_file": "train.src.txt",
                "train_labels_file": "train.trg.txt",
                "eval_features_file": "val.src.txt",
                "eval_labels_file": "val.trg.txt",
            },
            "eval": {
                "external_evaluators": "bleu",
                "steps": 1000,
                "early_stopping": {"metric": "bleu", "min_improvement": 0.2, "steps": 4},
            },
            "train": {
                "average_last_checkpoints": 0,
                "maximum_features_length": 150,
                "maximum_labels_length": 150,
            },
            "params": {
                "length_penalty": 0.2,
            },
        }
        if "max_step" in self._config:
            config["train"]["max_step"] = self._config["max_step"]

        return config

    def _get_parent_config(self, source_language_tag: str, target_language_tag: str) -> Optional[dict]:
        parents_dir = Path(self._config["parent_models_dir"])
        parent_model_dir = parents_dir / target_language_tag
        if not parent_model_dir.is_dir():
            return None
        return load_config(str(parent_model_dir / "config.yml"))


def _set_tf_log_level(log_level: int = logging.INFO) -> None:
    tf.get_logger().setLevel(log_level)
    # Do not display warnings from TensorFlow C++, because of spurious "PredictCost()" errors.
    # See https://github.com/tensorflow/tensorflow/issues/50575.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _convert_vocab(sp_vocab_path: Path, onmt_vocab_path: Path, tags: Set[str] = set()) -> None:
    special_tokens = [START_OF_SENTENCE_TOKEN, END_OF_SENTENCE_TOKEN, PADDING_TOKEN] + list(tags)

    vocab = Vocab(special_tokens)
    with sp_vocab_path.open("r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            token = line.rstrip("\r\n")
            index = token.rindex("\t")
            token = token[:index]
            if token in ("<unk>", "<s>", "</s>", "<blank>"):  # Ignore special tokens
                continue
            vocab.add(token)
    vocab.pad_to_multiple(8)
    vocab.serialize(onmt_vocab_path)

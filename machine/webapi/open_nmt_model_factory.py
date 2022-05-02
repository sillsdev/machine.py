import shutil
from pathlib import Path
from typing import Set

from opennmt import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN
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

    def init(self, engine_id: str) -> None:
        engine_dir = self._get_engine_dir(engine_id)
        engine_dir.mkdir(exist_ok=True)

    def create_model(self, engine_id: str) -> TranslationModel:
        model_config = self._create_model_config(engine_id)
        model_type: str = self._config["model"]
        mixed_precision: bool = self._config["mixed_precision"]
        return OpenNmtModel(model_type, model_config, mixed_precision=mixed_precision)

    def create_model_trainer(self, engine_id: str, corpus: ParallelTextCorpus) -> Trainer:
        engine_dir = self._get_engine_dir(engine_id)
        _convert_vocab(engine_dir / "src-sp.vocab", engine_dir / "src.vocab")
        _convert_vocab(engine_dir / "trg-sp.vocab", engine_dir / "trg.vocab")

        corpus = corpus.tokenize(self.create_source_tokenizer(engine_id), self.create_target_tokenizer(engine_id))

        # TODO: Add support for parent models
        model_config = self._create_model_config(engine_id)
        model_type: str = self._config["model"]
        mixed_precision: bool = self._config["mixed_precision"]
        return OpenNmtModelTrainer(
            model_type, model_config, corpus, mixed_precision=mixed_precision, use_temp_dir=False
        )

    def create_source_tokenizer(self, engine_id: str) -> Tokenizer[str, int, str]:
        return SentencePieceTokenizer(self._get_engine_dir(engine_id) / "src-sp.model")

    def create_source_tokenizer_trainer(self, engine_id: str, corpus: TextCorpus) -> Trainer:
        src_sp_model_prefix = self._get_engine_dir(engine_id) / "src-sp"
        return SentencePieceTrainer(
            corpus,
            vocab_size=8000,
            model_prefix=str(src_sp_model_prefix),
            normalization_rule_name="nmt_nfkc_cf",
        )

    def create_target_tokenizer(self, engine_id: str) -> Tokenizer[str, int, str]:
        return SentencePieceTokenizer(self._get_engine_dir(engine_id) / "trg-sp.model")

    def create_target_tokenizer_trainer(self, engine_id: str, corpus: TextCorpus) -> Trainer:
        trg_sp_model_prefix = self._get_engine_dir(engine_id) / "trg-sp"
        return SentencePieceTrainer(
            corpus,
            vocab_size=8000,
            model_prefix=str(trg_sp_model_prefix),
            normalization_rule_name="nmt_nfkc",
        )

    def create_target_detokenizer(self, engine_id: str) -> Detokenizer[str, str]:
        return SentencePieceDetokenizer()

    def cleanup(self, engine_id: str) -> None:
        shutil.rmtree(self._get_engine_dir(engine_id))

    def _create_model_config(self, engine_id: str) -> dict:
        return {
            "auto_config": True,
            "model_dir": str(self._get_engine_dir(engine_id)),
            "data": {
                "source_vocabulary": "src.vocab",
                "target_vocabulary": "trg.vocab",
                "train_features_file": "train.src.txt",
                "train_labels_file": "train.trg.txt",
                "eval_features_file": "val.src.txt",
                "eval_labels_file": "val.trg.txt",
            },
        }

    def _get_engine_dir(self, engine_id: str) -> Path:
        return Path(self._config["engines_dir"]) / engine_id


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

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Set

from bson.objectid import ObjectId
from opennmt import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN
from opennmt.data import Vocab
from pymongo.mongo_client import MongoClient
from sentencepiece import SentencePieceTrainer

from ..corpora.dictionary_text_corpus import DictionaryTextCorpus
from ..corpora.parallel_text_corpus import ParallelTextCorpus, flatten_parallel_text_corpora
from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_ref import TextFileRef
from ..scripture.verse_ref import VerseRef
from ..tokenization.sentencepiece import SentencePieceDetokenizer, SentencePieceTokenizer
from ..translation.tensorflow.open_nmt_model import OpenNmtModel
from ..translation.tensorflow.open_nmt_model_trainer import OpenNmtModelTrainer
from ..utils.canceled_error import CanceledError
from ..utils.progress_status import ProgressStatus
from .data_file_service import DataFileService
from .models import BUILD_STATE_ACTIVE, CORPUS_TYPE_SOURCE, CORPUS_TYPE_TARGET, Build, DataFile, Engine, Translation
from .repository import Repository

_TRANSLATION_INSERT_BUFFER_SIZE = 100


class BatchNmtEngineBuildJob:
    def __init__(
        self,
        engines: Repository[Engine],
        builds: Repository[Build],
        translations: Repository[Translation],
        data_file_service: DataFileService,
        config: dict,
    ) -> None:
        self._engines = engines
        self._builds = builds
        self._translations = translations
        self._data_file_service = data_file_service
        self._config = config

    def run(self, check_canceled: Optional[Callable[[], None]] = None) -> None:
        build_id: str = self._config["build"]
        engine_id: str = self._config["engine"]

        source_corpora = self._data_file_service.create_text_corpora(engine_id, CORPUS_TYPE_SOURCE)
        target_corpora = self._data_file_service.create_text_corpora(engine_id, CORPUS_TYPE_TARGET)
        parallel_corpora = _create_parallel_corpora(source_corpora, target_corpora)

        parallel_corpus = flatten_parallel_text_corpora(parallel_corpora.values())

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_sp_model_prefix = tmp_path / "src-sp"
            with parallel_corpus.get_rows() as rows:
                SentencePieceTrainer.Train(
                    sentence_iterator=(r.source_text for r in rows if len(r.source_segment) > 0),
                    vocab_size=8000,
                    model_prefix=str(src_sp_model_prefix),
                    normalization_rule_name="nmt_nfkc_cf",
                )
            _convert_vocab(src_sp_model_prefix.with_suffix(".vocab"), tmp_path / "src.vocab")

            if check_canceled is not None:
                check_canceled()

            source_tokenizer = SentencePieceTokenizer(src_sp_model_prefix.with_suffix(".model"))

            trg_sp_model_prefix = tmp_path / "trg-sp"
            with parallel_corpus.get_rows() as rows:
                SentencePieceTrainer.Train(
                    sentence_iterator=(r.target_text for r in rows if len(r.target_segment) > 0),
                    vocab_size=8000,
                    model_prefix=str(trg_sp_model_prefix),
                    normalization_rule_name="nmt_nfkc",
                )
            _convert_vocab(trg_sp_model_prefix.with_suffix(".vocab"), tmp_path / "trg.vocab")

            if check_canceled is not None:
                check_canceled()

            target_tokenizer = SentencePieceTokenizer(trg_sp_model_prefix.with_suffix(".model"))

            parallel_corpus = parallel_corpus.tokenize(source_tokenizer, target_tokenizer)

            model_type: str = self._config["model"]
            mixed_precision: bool = self._config["mixed_precision"]
            model_config = {
                "auto_config": True,
                "model_dir": str(tmp_path),
                "data": {
                    "source_vocabulary": "src.vocab",
                    "target_vocabulary": "trg.vocab",
                    "train_features_file": "train.src.txt",
                    "train_labels_file": "train.trg.txt",
                    "eval_features_file": "val.src.txt",
                    "eval_labels_file": "val.trg.txt",
                },
            }
            # TODO: Add support for parent models
            trainer = OpenNmtModelTrainer(
                model_type,
                model_config,
                parallel_corpus,
                mixed_precision=mixed_precision,
            )

            def update_progress(status: ProgressStatus) -> None:
                self._builds.update(
                    {"_id": ObjectId(build_id), "state": BUILD_STATE_ACTIVE},
                    {"$set": {"step": status.step, "message": status.message}},
                )

            trainer.train(update_progress, check_canceled)
            trainer.save()

            if check_canceled is not None:
                check_canceled()

            detokenizer = SentencePieceDetokenizer()
            model = OpenNmtModel(model_type, model_config, mixed_precision=mixed_precision)
            with model.create_engine() as translation_engine:
                for corpus_id, corpus in parallel_corpora.items():
                    if check_canceled is not None:
                        check_canceled()
                    corpus = corpus.filter(lambda r: len(r.target_segment) == 0)
                    with corpus.tokenize(source_tokenizer).get_rows() as rows:
                        translations = translation_engine.translate_batch(r.source_segment for r in rows)
                    with corpus.get_rows() as rows:
                        buffer: List[Translation] = []
                        for row, translation in zip(rows, translations):
                            refs = list(row.source_refs)
                            text_id = _get_text_id(refs[0])
                            text = detokenizer.detokenize(translation.target_segment)
                            buffer.append(
                                {
                                    "engineRef": ObjectId(engine_id),
                                    "corpusId": corpus_id,
                                    "textId": text_id,
                                    "refs": refs,
                                    "text": text,
                                }
                            )
                            if len(buffer) == _TRANSLATION_INSERT_BUFFER_SIZE:
                                self._translations.insert_many(buffer)
                                buffer.clear()
                        if len(buffer) > 0:
                            self._translations.insert_many(buffer)
                            buffer.clear()
            # TODO: save model to S3

            self._engines.update(
                engine_id,
                {
                    "$set": {
                        "confidence": round(trainer.stats.metrics["bleu"], 2),
                        "trainedSegmentCount": trainer.stats.trained_segment_count,
                    }
                },
            )


def _create_parallel_corpora(
    source_corpora: Dict[str, TextCorpus], target_corpora: Dict[str, TextCorpus]
) -> Dict[str, ParallelTextCorpus]:
    parallel_corpora: Dict[str, ParallelTextCorpus] = {}
    for source_id, source_corpus in source_corpora.items():
        target_corpus = target_corpora.get(source_id)
        if target_corpus is None:
            target_corpus = DictionaryTextCorpus()
        parallel_corpora[source_id] = source_corpus.align_rows(target_corpus, all_source_rows=True)

    return parallel_corpora


def _get_text_id(ref: Any) -> str:
    if isinstance(ref, VerseRef):
        return ref.book
    if isinstance(ref, TextFileRef):
        return ref.file_id
    raise ValueError(f"Unsupported ref type: {ref}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model and inferences on all untranslated data.")
    parser.add_argument("--engine", required=True, type=str, help="Engine Id")
    parser.add_argument("--build", required=True, type=str, help="Build Id")
    parser.add_argument("--data-files-dir", required=True, type=str, help="Data files directory")
    parser.add_argument("--mongo", required=True, type=str, help="Mongo server address")
    parser.add_argument("--database", required=True, type=str, help="Mongo database name")
    parser.add_argument("--cancellation-token-file", type=str, help="The cancellation token file")
    parser.add_argument("--model", type=str, default="TransformerBase", help="The NMT model")
    parser.add_argument(
        "--mixed-precision", default=False, action="store_true", help="Enables mixed precision training"
    )
    args = parser.parse_args()

    client = MongoClient(f"mongodb://{args.mongo}")
    database = client.get_database(args.database)
    engines: Repository[Engine] = Repository(database.engines)
    builds: Repository[Build] = Repository(database.builds, is_subscribable=True)
    data_files: Repository[DataFile] = Repository(database.files)
    translations: Repository[Translation] = Repository(database.translations)
    data_file_service = DataFileService(data_files, vars(args))
    job = BatchNmtEngineBuildJob(engines, builds, translations, data_file_service, vars(args))
    cancellation_token_file: Optional[str] = args.cancellation_token_file

    def check_canceled() -> None:
        if cancellation_token_file is not None and os.path.isfile(cancellation_token_file):
            raise CanceledError

    try:
        job.run(check_canceled)
    except CanceledError:
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional

from bson.objectid import ObjectId
from pymongo.mongo_client import MongoClient

from ..corpora.dictionary_text_corpus import DictionaryTextCorpus
from ..corpora.parallel_text_corpus import ParallelTextCorpus, flatten_parallel_text_corpora
from ..corpora.text_corpus import TextCorpus, flatten_text_corpora
from ..utils.canceled_error import CanceledError
from ..utils.progress_status import ProgressStatus
from .data_file_service import DataFileService
from .models import BUILD_STATE_ACTIVE, CORPUS_TYPE_SOURCE, CORPUS_TYPE_TARGET, Build, DataFile, Engine, Translation
from .nmt_model_factory import NmtModelFactory
from .open_nmt_model_factory import OpenNmtModelFactory
from .repository import Repository

_TRANSLATION_INSERT_BUFFER_SIZE = 100


class NmtEngineBuildJob:
    def __init__(
        self,
        engines: Repository[Engine],
        builds: Repository[Build],
        translations: Repository[Translation],
        data_file_service: DataFileService,
        nmt_model_factory: NmtModelFactory,
    ) -> None:
        self._engines = engines
        self._builds = builds
        self._translations = translations
        self._data_file_service = data_file_service
        self._nmt_model_factory = nmt_model_factory

    def run(self, engine_id: str, build_id: str, check_canceled: Optional[Callable[[], None]] = None) -> None:
        self._nmt_model_factory.init(engine_id)

        source_corpora = self._data_file_service.create_text_corpora(engine_id, CORPUS_TYPE_SOURCE)
        target_corpora = self._data_file_service.create_text_corpora(engine_id, CORPUS_TYPE_TARGET)
        parallel_corpora = _create_parallel_corpora(source_corpora, target_corpora)

        source_corpus = flatten_text_corpora(source_corpora.values())
        target_corpus = flatten_text_corpora(target_corpora.values())
        parallel_corpus = flatten_parallel_text_corpora(parallel_corpora.values())

        source_tokenizer_trainer = self._nmt_model_factory.create_source_tokenizer_trainer(engine_id, source_corpus)
        source_tokenizer_trainer.train()
        source_tokenizer_trainer.save()

        if check_canceled is not None:
            check_canceled()

        target_tokenizer_trainer = self._nmt_model_factory.create_target_tokenizer_trainer(engine_id, target_corpus)
        target_tokenizer_trainer.train()
        target_tokenizer_trainer.save()

        if check_canceled is not None:
            check_canceled()

        model_trainer = self._nmt_model_factory.create_model_trainer(engine_id, parallel_corpus)

        def update_progress(status: ProgressStatus) -> None:
            self._builds.update(
                {"_id": ObjectId(build_id), "state": BUILD_STATE_ACTIVE},
                {"$set": {"step": status.step, "message": status.message}},
            )

        model_trainer.train(update_progress, check_canceled)
        model_trainer.save()

        if check_canceled is not None:
            check_canceled()

        corpora_translate_text_ids = self._data_file_service.get_translate_text_ids(engine_id)
        source_tokenizer = self._nmt_model_factory.create_source_tokenizer(engine_id)
        target_detokenizer = self._nmt_model_factory.create_target_detokenizer(engine_id)
        with self._nmt_model_factory.create_model(engine_id) as model, model.create_engine() as translation_engine:
            for corpus_id, corpus in parallel_corpora.items():
                if check_canceled is not None:
                    check_canceled()
                translate_text_ids = corpora_translate_text_ids.get(corpus_id, set())
                corpus = corpus.filter(lambda r: len(r.target_segment) == 0 and r.text_id in translate_text_ids)
                with corpus.tokenize(source_tokenizer).get_rows() as rows:
                    translations = translation_engine.translate_batch(r.source_segment for r in rows)
                with corpus.get_rows() as rows:
                    buffer: List[Translation] = []
                    for row, translation in zip(rows, translations):
                        refs = list(row.source_refs)
                        text = target_detokenizer.detokenize(translation.target_segment)
                        buffer.append(
                            {
                                "engineRef": ObjectId(engine_id),
                                "corpusId": corpus_id,
                                "textId": row.text_id,
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
                    "confidence": round(model_trainer.stats.metrics["bleu"], 2),
                    "trainedSegmentCount": model_trainer.stats.train_size,
                }
            },
        )

        self._nmt_model_factory.cleanup(engine_id)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--engine", required=True, type=str, help="Engine Id")
    parser.add_argument("--build", required=True, type=str, help="Build Id")
    parser.add_argument("--engines-dir", required=True, type=str, help="Engines directory")
    parser.add_argument("--data-files-dir", required=True, type=str, help="Data files directory")
    parser.add_argument("--mongo", required=True, type=str, help="Mongo server address")
    parser.add_argument("--database", required=True, type=str, help="Mongo database name")
    parser.add_argument("--cancellation-token-file", type=str, help="The cancellation token file")
    parser.add_argument("--model", type=str, default="TransformerBase", help="The NMT model")
    parser.add_argument(
        "--mixed-precision", default=False, action="store_true", help="Enables mixed precision training"
    )
    args = parser.parse_args()

    config = vars(args)
    client = MongoClient(f"mongodb://{args.mongo}")
    database = client.get_database(args.database)
    engines: Repository[Engine] = Repository(database.engines)
    builds: Repository[Build] = Repository(database.builds, is_subscribable=True)
    data_files: Repository[DataFile] = Repository(database.files)
    translations: Repository[Translation] = Repository(database.translations)
    data_file_service = DataFileService(data_files, config)
    nmt_model_factory = OpenNmtModelFactory(config)
    job = NmtEngineBuildJob(engines, builds, translations, data_file_service, nmt_model_factory)
    cancellation_token_file: Optional[str] = args.cancellation_token_file

    def check_canceled() -> None:
        if cancellation_token_file is not None and os.path.isfile(cancellation_token_file):
            raise CanceledError

    try:
        job.run(args.engine, args.build, check_canceled)
    except CanceledError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

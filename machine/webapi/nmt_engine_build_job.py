import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Set

from bson.objectid import ObjectId
from pymongo.mongo_client import MongoClient

from ..corpora.parallel_text_corpus import ParallelTextCorpus, flatten_parallel_text_corpora
from ..corpora.text_corpus import TextCorpus, flatten_text_corpora
from ..translation.translation_engine import translate_corpus
from ..utils.canceled_error import CanceledError
from ..utils.progress_status import ProgressStatus
from .corpus_service import CorpusService
from .models import BUILD_STATE_ACTIVE, Build, Corpus, Pretranslation, TranslationEngine
from .nmt_model_factory import NmtModelFactory
from .open_nmt_model_factory import OpenNmtModelFactory
from .repository import Repository

_PRETRANSLATION_INSERT_BUFFER_SIZE = 100


class NmtEngineBuildJob:
    def __init__(
        self,
        engines: Repository[TranslationEngine],
        builds: Repository[Build],
        pretranslations: Repository[Pretranslation],
        corpus_service: CorpusService,
        nmt_model_factory: NmtModelFactory,
    ) -> None:
        self._engines = engines
        self._builds = builds
        self._pretranslations = pretranslations
        self._corpus_service = corpus_service
        self._nmt_model_factory = nmt_model_factory

    def run(self, engine_id: str, build_id: str, check_canceled: Optional[Callable[[], None]] = None) -> None:
        engine = self._engines.get(engine_id)
        if engine is None:
            return

        self._nmt_model_factory.init(engine_id)

        source_corpora: List[TextCorpus] = []
        target_corpora: List[TextCorpus] = []
        parallel_corpora: Dict[str, ParallelTextCorpus] = {}
        pretranslate_corpora: Set[str] = set()
        for corpus in engine["corpora"]:
            corpus_id = str(corpus["corpusRef"])
            sc = self._corpus_service.create_text_corpus(corpus_id, engine["sourceLanguageTag"])
            if sc is not None:
                source_corpora.append(sc)
            tc = self._corpus_service.create_text_corpus(corpus_id, engine["targetLanguageTag"])
            if tc is not None:
                target_corpora.append(tc)

            if sc is not None and tc is not None:
                parallel_corpora[corpus_id] = sc.align_rows(tc, all_source_rows=True)

            if corpus["pretranslate"]:
                pretranslate_corpora.add(corpus_id)

        source_corpus = flatten_text_corpora(source_corpora)
        target_corpus = flatten_text_corpora(target_corpora)
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

        model_trainer = self._nmt_model_factory.create_model_trainer(
            engine_id, engine["sourceLanguageTag"], engine["targetLanguageTag"], parallel_corpus
        )

        def update_progress(status: ProgressStatus) -> None:
            self._builds.update(
                {"_id": ObjectId(build_id), "state": BUILD_STATE_ACTIVE},
                {"$set": {"step": status.step, "message": status.message}},
            )

        model_trainer.train(update_progress, check_canceled)
        model_trainer.save()

        if check_canceled is not None:
            check_canceled()

        source_tokenizer = self._nmt_model_factory.create_source_tokenizer(engine_id)
        target_detokenizer = self._nmt_model_factory.create_target_detokenizer(engine_id)
        with self._nmt_model_factory.create_model(engine_id) as model, model.create_engine() as translation_engine:
            for corpus_id, corpus in parallel_corpora.items():
                if corpus_id not in pretranslate_corpora:
                    continue
                if check_canceled is not None:
                    check_canceled()
                corpus = corpus.filter(lambda r: len(r.target_refs) > 0 and len(r.target_segment) == 0)
                corpus = corpus.tokenize_source(source_tokenizer)
                corpus = translate_corpus(translation_engine, corpus)
                corpus = corpus.detokenize_target(target_detokenizer)
                with corpus.batch(_PRETRANSLATION_INSERT_BUFFER_SIZE) as batches:
                    for batch in batches:
                        self._pretranslations.insert_all(
                            [
                                {
                                    "translationEngineRef": ObjectId(engine_id),
                                    "corpusRef": ObjectId(corpus_id),
                                    "textId": row.text_id,
                                    "refs": [str(r) for r in row.target_refs],
                                    "translation": row.target_text,
                                }
                                for row in batch
                            ]
                        )

        self._nmt_model_factory.save_model(engine_id)
        self._engines.update(
            engine_id,
            {
                "$set": {
                    "confidence": round(model_trainer.stats.metrics["bleu"], 2),
                    "trainSize": model_trainer.stats.train_size,
                }
            },
        )

        self._nmt_model_factory.cleanup(engine_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--engine", required=True, type=str, help="Engine Id")
    parser.add_argument("--build", required=True, type=str, help="Build Id")
    parser.add_argument("--models-dir", required=True, type=str, help="Models directory")
    parser.add_argument("--data-files-dir", required=True, type=str, help="Data files directory")
    parser.add_argument("--parent-models-dir", required=True, type=str, help="Parent models directory")
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
    engines: Repository[TranslationEngine] = Repository(database.engines)
    builds: Repository[Build] = Repository(database.builds, is_subscribable=True)
    corpora: Repository[Corpus] = Repository(database.corpora)
    pretranslations: Repository[Pretranslation] = Repository(database.pretranslations)
    corpus_service = CorpusService(corpora, config)
    nmt_model_factory = OpenNmtModelFactory(config)
    job = NmtEngineBuildJob(engines, builds, pretranslations, corpus_service, nmt_model_factory)
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

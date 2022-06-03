from typing import Type, TypeVar, cast

import pytest
from bson.objectid import ObjectId
from mockito import ANY, mock, verify, when
from mongomock.mongo_client import MongoClient

import machine.webapi.models as m
from machine.annotations import Range
from machine.corpora import DictionaryTextCorpus, MemoryText, TextFileRef, TextRow
from machine.tokenization import LatinWordDetokenizer, WhitespaceTokenizer
from machine.translation import (
    Phrase,
    Trainer,
    TrainStats,
    TranslationEngine,
    TranslationModel,
    TranslationResult,
    TranslationSources,
    WordAlignmentMatrix,
)
from machine.utils import CanceledError
from machine.webapi.corpus_service import CorpusService
from machine.webapi.nmt_engine_build_job import NmtEngineBuildJob
from machine.webapi.nmt_model_factory import NmtModelFactory
from machine.webapi.repository import Repository


def test_run() -> None:
    env = _TestEnvironment()
    env.job.run(str(env.engine_id), str(env.build_id))

    engine = env.engines.get(env.engine_id)
    assert engine is not None
    assert engine["confidence"] == 30.0
    assert engine["trainSize"] == 3

    verify(env.engine, times=1).translate_batch(...)
    pretranslations = list(env.pretranslations.get_all({"translationEngineRef": env.engine_id}))
    assert len(pretranslations) == 1
    assert pretranslations[0]["translation"] == "Please, I have booked a room."


def test_cancel() -> None:
    env = _TestEnvironment()
    checker = _CancellationChecker(3)
    with pytest.raises(CanceledError):
        env.job.run(str(env.engine_id), str(env.build_id), checker.check_canceled)

    engine = env.engines.get(env.engine_id)
    assert engine is not None
    assert engine["confidence"] == 0
    assert engine["trainSize"] == 0

    pretranslations = list(env.pretranslations.get_all({"translationEngineRef": env.engine_id}))
    assert len(pretranslations) == 0


class _TestEnvironment:
    def __init__(self) -> None:
        client = MongoClient()
        self.engines: Repository[m.TranslationEngine] = Repository(client.machine.engines)
        corpus1_id = ObjectId()
        corpus2_id = ObjectId()
        self.engine_id = self.engines.insert(
            m.TranslationEngine(
                sourceLanguageTag="es",
                targetLanguageTag="en",
                type=m.ENGINE_TYPE_NMT,
                owner="app1",
                corpora=[
                    m.TranslationEngineCorpus(corpusRef=corpus1_id, pretranslate=False),
                    m.TranslationEngineCorpus(corpusRef=corpus2_id, pretranslate=True),
                ],
                isBuilding=False,
                modelRevision=0,
                confidence=0,
                trainSize=0,
            )
        )
        self.builds: Repository[m.Build] = Repository(client.machine.builds)
        self.build_id = self.builds.insert(
            m.Build(parentRef=self.engine_id, jobId="job1", step=0, state=m.BUILD_STATE_PENDING)
        )
        self.pretranslations: Repository[m.Pretranslation] = Repository(client.machine.pretranslations)

        self.corpus_service = _mock(CorpusService)
        when(self.corpus_service).create_text_corpus(str(corpus1_id), "es").thenReturn(
            DictionaryTextCorpus(
                MemoryText(
                    "text1",
                    [
                        _row("text1", 1, "¿ Le importaría darnos las llaves de la habitación , por favor ?"),
                        _row(
                            "text1",
                            2,
                            "He hecho la reserva de una habitación tranquila doble con teléfono y televisión a "
                            "nombre de Rosario Cabedo .",
                        ),
                        _row("text1", 3, "¿ Le importaría cambiarme a otra habitación más tranquila ?"),
                        _row("text1", 4, "Me parece que existe un problema ."),
                    ],
                )
            )
        )
        when(self.corpus_service).create_text_corpus(str(corpus2_id), "es").thenReturn(
            DictionaryTextCorpus(
                MemoryText("text2", [_row("text2", 1, "Por favor , tengo reservada una habitación .")]),
            )
        )
        when(self.corpus_service).create_text_corpus(str(corpus1_id), "en").thenReturn(
            DictionaryTextCorpus(
                MemoryText(
                    "text1",
                    [
                        _row("text1", 1, "Would you mind giving us the keys to the room , please ?"),
                        _row(
                            "text1",
                            2,
                            "I have made a reservation for a quiet , double room with a telephone and a tv for "
                            "Rosario Cabedo .",
                        ),
                        _row("text1", 3, "Would you mind moving me to a quieter room ?"),
                        _row("text1", 4, ""),
                    ],
                ),
            )
        )
        when(self.corpus_service).create_text_corpus(str(corpus2_id), "en").thenReturn(
            DictionaryTextCorpus(MemoryText("text2", [_row("text2", 1, "")]))
        )

        self.source_tokenizer_trainer = _mock(Trainer)
        when(self.source_tokenizer_trainer).train().thenReturn()
        when(self.source_tokenizer_trainer).save().thenReturn()

        self.target_tokenizer_trainer = _mock(Trainer)
        when(self.target_tokenizer_trainer).train().thenReturn()
        when(self.target_tokenizer_trainer).save().thenReturn()

        self.model_trainer = _mock(Trainer)
        when(self.model_trainer).train(ANY, ANY).thenReturn()
        when(self.model_trainer).save().thenReturn()
        stats = TrainStats()
        stats.train_size = 3
        stats.metrics["bleu"] = 30.0
        setattr(self.model_trainer, "stats", stats)

        self.engine = _mock(TranslationEngine)
        when(self.engine).translate_batch(ANY).thenReturn(
            [
                TranslationResult(
                    source_segment_length=8,
                    target_segment="Please , I have booked a room .".split(),
                    word_confidences=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    word_sources=[
                        TranslationSources.NMT,
                        TranslationSources.NMT,
                        TranslationSources.NMT,
                        TranslationSources.NMT,
                        TranslationSources.NMT,
                        TranslationSources.NMT,
                        TranslationSources.NMT,
                        TranslationSources.NMT,
                    ],
                    alignment=WordAlignmentMatrix.from_word_pairs(
                        8, 8, {(0, 0), (1, 0), (2, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)}
                    ),
                    phrases=[Phrase(Range.create(0, 8), 8, 0.5)],
                )
            ]
        )

        self.model = _mock(TranslationModel)
        when(self.model).create_engine().thenReturn(self.engine)

        self.nmt_model_factory = _mock(NmtModelFactory)
        when(self.nmt_model_factory).init(ANY).thenReturn()
        when(self.nmt_model_factory).create_source_tokenizer_trainer(ANY, ANY).thenReturn(self.source_tokenizer_trainer)
        when(self.nmt_model_factory).create_target_tokenizer_trainer(ANY, ANY).thenReturn(self.target_tokenizer_trainer)
        when(self.nmt_model_factory).create_model_trainer(ANY, ANY, ANY, ANY).thenReturn(self.model_trainer)
        when(self.nmt_model_factory).create_source_tokenizer(ANY).thenReturn(WhitespaceTokenizer())
        when(self.nmt_model_factory).create_target_detokenizer(ANY).thenReturn(LatinWordDetokenizer())
        when(self.nmt_model_factory).create_model(ANY).thenReturn(self.model)
        when(self.nmt_model_factory).save_model(ANY).thenReturn()
        when(self.nmt_model_factory).cleanup(ANY).thenReturn()

        self.job = NmtEngineBuildJob(
            self.engines, self.builds, self.pretranslations, self.corpus_service, self.nmt_model_factory
        )


def _row(text_id: str, line: int, text: str) -> TextRow:
    return TextRow(text_id, TextFileRef(text_id, line), [text] if len(text) > 0 else [], is_empty=len(text) == 0)


T = TypeVar("T")


def _mock(class_to_mock: Type[T]) -> T:
    o = cast(T, mock(class_to_mock))
    if hasattr(class_to_mock, "__enter__"):
        when(o).__enter__().thenReturn(o)
        when(o).__exit__(ANY, ANY, ANY).thenReturn()
    return o


class _CancellationChecker:
    def __init__(self, raise_count: int) -> None:
        self._call_count = 0
        self._raise_count = raise_count

    def check_canceled(self) -> None:
        self._call_count += 1
        if self._call_count == self._raise_count:
            raise CanceledError

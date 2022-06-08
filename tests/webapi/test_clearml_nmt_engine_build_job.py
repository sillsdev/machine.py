import json
from contextlib import contextmanager
from io import StringIO
from typing import Iterator, Type, TypeVar, cast

import pytest
from mockito import ANY, mock, verify, when

from machine.annotations import Range
from machine.corpora import DictionaryTextCorpus
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
from machine.utils import CanceledError, ContextManagedGenerator
from machine.webapi.clearml_nmt_engine_build_job import ClearMLNmtEngineBuildJob
from machine.webapi.nmt_model_factory import NmtModelFactory
from machine.webapi.shared_file_service import PretranslationInfo, PretranslationWriter, SharedFileService


def test_run() -> None:
    env = _TestEnvironment()
    env.job.run()

    verify(env.engine, times=1).translate_batch(...)
    pretranslations = json.loads(env.target_pretranslations)
    assert len(pretranslations) == 1
    assert pretranslations[0]["segment"] == "Please, I have booked a room."


def test_cancel() -> None:
    env = _TestEnvironment()
    checker = _CancellationChecker(3)
    with pytest.raises(CanceledError):
        env.job.run(checker.check_canceled)

    assert env.target_pretranslations == ""


class _TestEnvironment:
    def __init__(self) -> None:
        config = {"src_lang": "es", "trg_lang": "en"}

        self.source_tokenizer_trainer = _mock(Trainer)
        when(self.source_tokenizer_trainer).train().thenReturn()
        when(self.source_tokenizer_trainer).save().thenReturn()

        self.target_tokenizer_trainer = _mock(Trainer)
        when(self.target_tokenizer_trainer).train().thenReturn()
        when(self.target_tokenizer_trainer).save().thenReturn()

        self.model_trainer = _mock(Trainer)
        when(self.model_trainer).train(check_canceled=ANY).thenReturn()
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
        when(self.nmt_model_factory).init().thenReturn()
        when(self.nmt_model_factory).create_source_tokenizer_trainer(ANY).thenReturn(self.source_tokenizer_trainer)
        when(self.nmt_model_factory).create_target_tokenizer_trainer(ANY).thenReturn(self.target_tokenizer_trainer)
        when(self.nmt_model_factory).create_model_trainer(ANY, ANY, ANY).thenReturn(self.model_trainer)
        when(self.nmt_model_factory).create_source_tokenizer().thenReturn(WhitespaceTokenizer())
        when(self.nmt_model_factory).create_target_detokenizer().thenReturn(LatinWordDetokenizer())
        when(self.nmt_model_factory).create_model().thenReturn(self.model)

        self.shared_file_service = _mock(SharedFileService)
        when(self.shared_file_service).init().thenReturn()
        when(self.shared_file_service).create_source_corpus().thenReturn(DictionaryTextCorpus())
        when(self.shared_file_service).create_target_corpus().thenReturn(DictionaryTextCorpus())
        when(self.shared_file_service).get_source_pretranslations().thenReturn(
            ContextManagedGenerator(
                (
                    pi
                    for pi in [
                        PretranslationInfo(
                            corpusId="corpus1",
                            textId="text1",
                            refs=["ref1"],
                            segment="Por favor , tengo reservada una habitación .",
                        )
                    ]
                )
            )
        )

        self.target_pretranslations = ""

        @contextmanager
        def open_target_pretranslation_writer(env: _TestEnvironment) -> Iterator[PretranslationWriter]:
            file = StringIO()
            file.write("[\n")
            yield PretranslationWriter(file)
            file.write("\n]\n")
            env.target_pretranslations = file.getvalue()

        when(self.shared_file_service).open_target_pretranslation_writer().thenReturn(
            open_target_pretranslation_writer(self)
        )

        self.job = ClearMLNmtEngineBuildJob(config, self.nmt_model_factory, self.shared_file_service)


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

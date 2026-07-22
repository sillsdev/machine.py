import json
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Iterator

from decoy import Decoy, matchers
from pytest import raises
from testutils.mock_settings import MockSettings

from machine.annotations import Range
from machine.corpora import DictionaryTextCorpus
from machine.jobs import (
    DictToJsonWriter,
    NmtEngineBuildJob,
    NmtModelFactory,
    PretranslationInfo,
    TranslationFileService,
    WordAlignmentModelFactory,
)
from machine.translation import (
    Phrase,
    Trainer,
    TrainStats,
    TranslationEngine,
    TranslationResult,
    TranslationSources,
    WordAlignmentMatrix,
    WordAlignmentModel,
)
from machine.utils import CanceledError, ContextManagedGenerator


def test_run(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    env.job.run()

    pretranslations = json.loads(env.target_pretranslations)
    assert len(pretranslations) == 1
    assert pretranslations[0]["translation"] == "Please, I have booked a room."
    assert pretranslations[0]["sourceTokens"] == [
        "Por",
        "favor",
        ",",
        "tengo",
        "reservada",
        "una",
        "habitación",
        ".",
    ]
    assert pretranslations[0]["translationTokens"] == ["Please", ",", "I", "have", "booked", "a", "room", "."]
    assert len(pretranslations[0]["alignment"]) > 0
    assert pretranslations[0]["sequenceConfidence"] == 0.5
    decoy.verify(env.translation_file_service.save_model(Path("model.tar.gz"), "models/save-model.tar.gz"), times=1)


def test_cancel(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    checker = _CancellationChecker(3)
    with raises(CanceledError):
        env.job.run(check_canceled=checker.check_canceled)

    assert env.target_pretranslations == ""


class _TestEnvironment:
    def __init__(self, decoy: Decoy) -> None:
        self.source_tokenizer_trainer = decoy.mock(cls=Trainer)
        self.target_tokenizer_trainer = decoy.mock(cls=Trainer)

        self.model_trainer = decoy.mock(cls=Trainer)
        stats = TrainStats()
        stats.train_corpus_size = 3
        stats.metrics["bleu"] = 30.0
        decoy.when(self.model_trainer.stats).then_return(stats)

        self.engine = decoy.mock(cls=TranslationEngine)
        decoy.when(self.engine.__enter__()).then_return(self.engine)
        decoy.when(self.engine.translate_batch(matchers.Anything())).then_return(
            [
                TranslationResult(
                    translation="Please, I have booked a room.",
                    source_tokens="Por favor , tengo reservada una habitación .".split(),
                    target_tokens="Please , I have booked a room .".split(),
                    confidences=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    sequence_confidence=0.5,
                    sources=[
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
                    phrases=[Phrase(Range.create(0, 8), 8)],
                )
            ]
        )

        self.nmt_model_factory = decoy.mock(cls=NmtModelFactory)
        decoy.when(self.nmt_model_factory.train_tokenizer).then_return(True)
        decoy.when(self.nmt_model_factory.create_source_tokenizer_trainer(matchers.Anything())).then_return(
            self.source_tokenizer_trainer
        )
        decoy.when(self.nmt_model_factory.create_target_tokenizer_trainer(matchers.Anything())).then_return(
            self.target_tokenizer_trainer
        )
        decoy.when(self.nmt_model_factory.create_model_trainer(matchers.Anything())).then_return(self.model_trainer)
        decoy.when(self.nmt_model_factory.create_engine()).then_return(self.engine)
        decoy.when(self.nmt_model_factory.save_model()).then_return(Path("model.tar.gz"))

        self.translation_file_service = decoy.mock(cls=TranslationFileService)
        decoy.when(self.translation_file_service.create_source_corpus()).then_return(DictionaryTextCorpus())
        decoy.when(self.translation_file_service.create_target_corpus()).then_return(DictionaryTextCorpus())
        decoy.when(self.translation_file_service.exists_source_corpus()).then_return(True)
        decoy.when(self.translation_file_service.exists_target_corpus()).then_return(True)

        decoy.when(self.translation_file_service.get_source_pretranslations()).then_do(
            lambda: ContextManagedGenerator(
                (
                    pi
                    for pi in [
                        PretranslationInfo(
                            corpusId="corpus1",
                            textId="text1",
                            sourceRefs=["ref1"],
                            targetRefs=["ref1"],
                            translation="Por favor, tengo reservada una habitación.",
                            sourceTokens=[],
                            translationTokens=[],
                            alignment="",
                            sequenceConfidence=0.5,
                        )
                    ]
                )
            )
        )

        self.target_pretranslations = ""

        @contextmanager
        def open_target_pretranslation_writer(env: _TestEnvironment) -> Iterator[DictToJsonWriter]:
            file = StringIO()
            file.write("[\n")
            yield DictToJsonWriter(file)
            file.write("\n]\n")
            env.target_pretranslations = file.getvalue()

        decoy.when(self.translation_file_service.open_target_pretranslation_writer()).then_do(
            lambda: open_target_pretranslation_writer(self)
        )

        self.alignment_model_trainer = decoy.mock(cls=Trainer)
        decoy.when(self.alignment_model_trainer.__enter__()).then_return(self.alignment_model_trainer)
        stats = TrainStats()
        decoy.when(self.alignment_model_trainer.stats).then_return(stats)

        self.model = decoy.mock(cls=WordAlignmentModel)
        decoy.when(self.model.__enter__()).then_return(self.model)
        decoy.when(self.model.align_batch(matchers.Anything())).then_return(
            [
                WordAlignmentMatrix.from_word_pairs(
                    row_count=8,
                    column_count=8,
                    set_values=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
                ),
            ]
        )

        self.word_alignment_model_factory = decoy.mock(cls=WordAlignmentModelFactory)
        decoy.when(
            self.word_alignment_model_factory.create_model_trainer(matchers.Anything(), matchers.Anything())
        ).then_return(self.alignment_model_trainer)
        decoy.when(self.word_alignment_model_factory.create_alignment_model()).then_return(self.model)
        decoy.when(self.word_alignment_model_factory.save_model()).then_return(Path("model.zip"))

        self.job = NmtEngineBuildJob(
            MockSettings(
                {
                    "src_lang": "es",
                    "trg_lang": "en",
                    "save_model": "save-model",
                    "inference_batch_size": 100,
                    "align_pretranslations": True,
                    "build_id": "my_build",
                    "thot_align": {
                        "tokenizer": "latin",
                        "model_type": "eflomal",
                        "word_alignment_heuristic": "grow-diag-final-and",
                    },
                    "data_dir": "~/machine",
                    "shared_file_folder": "dev",
                }
            ),
            self.nmt_model_factory,
            self.translation_file_service,
            self.word_alignment_model_factory,
        )


class _CancellationChecker:
    def __init__(self, raise_count: int) -> None:
        self._call_count = 0
        self._raise_count = raise_count

    def check_canceled(self) -> None:
        self._call_count += 1
        if self._call_count == self._raise_count:
            raise CanceledError

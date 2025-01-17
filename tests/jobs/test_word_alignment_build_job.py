import json
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Iterator

from decoy import Decoy, matchers
from pytest import raises
from testutils.mock_settings import MockSettings

from machine.corpora import DictionaryTextCorpus, MemoryText, TextRow
from machine.jobs import DictToJsonWriter, WordAlignmentBuildJob, WordAlignmentModelFactory
from machine.jobs.word_alignment_file_service import WordAlignmentFileService, WordAlignmentInput
from machine.translation import Trainer, TrainStats, WordAlignmentMatrix
from machine.translation.word_alignment_model import WordAlignmentModel
from machine.utils import CanceledError


def test_run(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    env.job.run()

    alignments = json.loads(env.alignment_json)
    assert len(alignments) == 1
    assert alignments[0]["alignment"] == "0-0 1-1 2-2"
    decoy.verify(
        env.word_alignment_file_service.save_model(matchers.Anything(), f"builds/{env.job._config.build_id}/model.zip"),
        times=1,
    )


def test_cancel(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    checker = _CancellationChecker(3)
    with raises(CanceledError):
        env.job.run(check_canceled=checker.check_canceled)

    assert env.alignment_json == ""


class _TestEnvironment:
    def __init__(self, decoy: Decoy) -> None:
        self.model_trainer = decoy.mock(cls=Trainer)
        decoy.when(self.model_trainer.__enter__()).then_return(self.model_trainer)
        stats = TrainStats()
        stats.train_corpus_size = 3
        stats.metrics["bleu"] = 30.0
        decoy.when(self.model_trainer.stats).then_return(stats)

        self.model = decoy.mock(cls=WordAlignmentModel)
        decoy.when(self.model.__enter__()).then_return(self.model)
        decoy.when(self.model.align_batch(matchers.Anything())).then_return(
            [
                WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
            ]
        )

        self.word_alignment_model_factory = decoy.mock(cls=WordAlignmentModelFactory)
        decoy.when(
            self.word_alignment_model_factory.create_model_trainer(matchers.Anything(), matchers.Anything())
        ).then_return(self.model_trainer)
        decoy.when(self.word_alignment_model_factory.create_alignment_model()).then_return(self.model)
        decoy.when(self.word_alignment_model_factory.save_model()).then_return(Path("model.zip"))

        self.word_alignment_file_service = decoy.mock(cls=WordAlignmentFileService)
        decoy.when(self.word_alignment_file_service.create_source_corpus()).then_return(
            DictionaryTextCorpus(
                MemoryText(
                    "text1",
                    [
                        TextRow("text1", 1, ["¿Le importaría darnos las llaves de la habitación, por favor?"]),
                        TextRow("text1", 2, ["¿Le importaría cambiarme a otra habitación más tranquila?"]),
                        TextRow("text1", 3, ["Me parece que existe un problema."]),
                    ],
                )
            )
        )
        decoy.when(self.word_alignment_file_service.create_target_corpus()).then_return(
            DictionaryTextCorpus(
                MemoryText(
                    "text1",
                    [
                        TextRow("text1", 1, ["Would you mind giving us the room keys, please?"]),
                        TextRow("text1", 2, ["Would you mind moving me to another quieter room?"]),
                        TextRow("text1", 3, ["I think there is a problem."]),
                    ],
                )
            )
        )
        decoy.when(self.word_alignment_file_service.exists_source_corpus()).then_return(True)
        decoy.when(self.word_alignment_file_service.exists_target_corpus()).then_return(True)

        decoy.when(self.word_alignment_file_service.get_word_alignment_inputs()).then_return(
            [
                WordAlignmentInput(
                    corpusId="corpus1",
                    textId="text1",
                    refs=["1"],
                    source="¿Le importaría darnos las llaves de la habitación, por favor?",
                    target="Would you mind giving us the room keys, please?",
                ),
                WordAlignmentInput(
                    corpusId="corpus1",
                    textId="text1",
                    refs=["2"],
                    source="¿Le importaría cambiarme a otra habitación más tranquila?",
                    target="Would you mind moving me to another quieter room?",
                ),
                WordAlignmentInput(
                    corpusId="corpus1",
                    textId="text1",
                    refs=["3"],
                    source="Me parece que existe un problema.",
                    target="I think there is a problem.",
                ),
            ]
        )

        self.alignment_json = ""

        @contextmanager
        def open_target_alignment_writer(env: _TestEnvironment) -> Iterator[DictToJsonWriter]:
            file = StringIO()
            file.write("[\n")
            yield DictToJsonWriter(file)
            file.write("\n]\n")
            env.alignment_json = file.getvalue()

        decoy.when(self.word_alignment_file_service.open_alignment_output_writer()).then_do(
            lambda: open_target_alignment_writer(self)
        )

        self.job = WordAlignmentBuildJob(
            MockSettings({"build_id": "mybuild", "inference_batch_size": 100, "thot_align": {"tokenizer": "latin"}}),
            self.word_alignment_model_factory,
            self.word_alignment_file_service,
        )


class _CancellationChecker:
    def __init__(self, raise_count: int) -> None:
        self._call_count = 0
        self._raise_count = raise_count

    def check_canceled(self) -> None:
        self._call_count += 1
        if self._call_count == self._raise_count:
            raise CanceledError

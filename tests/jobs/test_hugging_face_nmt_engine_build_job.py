import json
import os
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Iterator, List
from unittest.mock import patch

from decoy import Decoy, matchers
from pytest import raises
from testutils.mock_llm_completion_client import MockLlmCompletionClient
from testutils.mock_settings import MockSettings

from machine.annotations import Range
from machine.corpora import DictionaryTextCorpus, MemoryText, TextRow
from machine.jobs import (
    DictToJsonWriter,
    HuggingFaceNmtEngineBuildJob,
    NmtModelFactory,
    PretranslationInfo,
    TranslationFileService,
    WordAlignmentModelFactory,
)
from machine.jobs.remote_llm_engine_build_job import RemoteLlmEngineBuildJob
from machine.jobs.remote_llm_model_factory import RemoteLlmModelFactory
from machine.translation import (
    Phrase,
    Trainer,
    TrainStats,
    TransductiveWordAlignmentModel,
    TranslationEngine,
    TranslationResult,
    TranslationSources,
    WordAlignmentMatrix,
    WordAlignmentModel,
)
from machine.translation.llm import Completion, ContentBlockError, ExampleRetrievalMethod, RefusalError
from machine.utils import CanceledError, ContextManagedGenerator


def test_run(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy, transductive=True)
    stages = []
    env.job.run(progress=lambda status: stages.append(getattr(status, "phase_stage", None)))

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
    assert stages.index("inference") < stages.index("align")
    decoy.verify(env.model.align_batch(matchers.Anything()), times=0)
    decoy.verify(env.translation_file_service.save_model(Path("model.tar.gz"), "models/save-model.tar.gz"), times=1)


def test_run_with_surrogate_pair(decoy: Decoy) -> None:
    env = _TestEnvironment(
        decoy,
        transductive=True,
        source_text="亚撒黑仍不肯转开。故此，押尼珥就用枪\ud863\udc54刺入他的肚腹，甚至枪从背后透出，亚撒黑就在那里仆倒而死。众人赶到亚撒黑仆倒而死的地方，就都站住。",
    )
    env.job.run()

    pretranslations = json.loads(env.target_pretranslations)
    assert len(pretranslations) == 1
    assert pretranslations[0]["translation"] == "Please, I have booked a room."
    assert pretranslations[0]["sourceTokens"] == [
        "亚撒黑仍不肯转开",
        "。",
        "故此",
        "，",
        "押尼珥就用枪𨱔刺入他的肚腹",
        "，",
        "甚至枪从背后透出",
        "，",
        "亚撒黑就在那里仆倒而死",
        "。",
        "众人赶到亚撒黑仆倒而死的地方",
        "，",
        "就都站住",
        "。",
    ]
    assert pretranslations[0]["translationTokens"] == ["Please", ",", "I", "have", "booked", "a", "room", "."]
    assert len(pretranslations[0]["alignment"]) > 0
    assert pretranslations[0]["sequenceConfidence"] == 0.5
    decoy.verify(env.model.align_batch(matchers.Anything()), times=0)
    decoy.verify(env.translation_file_service.save_model(Path("model.tar.gz"), "models/save-model.tar.gz"), times=1)


def test_run_with_lone_surrogate(decoy: Decoy) -> None:
    env = _TestEnvironment(
        decoy,
        transductive=True,
        source_text="lone\ud863surrogate",
    )
    env.job.run()

    pretranslations = json.loads(env.target_pretranslations)
    assert len(pretranslations) == 1
    assert pretranslations[0]["sourceTokens"] == ["lonesurrogate"]


def test_run_inductive_fallback(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy, transductive=False)
    env.job.run()

    pretranslations = json.loads(env.target_pretranslations)
    assert len(pretranslations) == 1
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


def test_run_batched_transductive(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy, transductive=True, num_pretranslations=3, batch_size=2)
    env.job.run()

    pretranslations = json.loads(env.target_pretranslations)
    assert len(pretranslations) == 3
    assert [pt["textId"] for pt in pretranslations] == ["text1", "text2", "text3"]
    for pretranslation in pretranslations:
        assert pretranslation["translation"] == "Please, I have booked a room."
        assert len(pretranslation["alignment"]) > 0
    assert env.translate_batch_sizes == [2, 1]
    assert env.training_alignment_requests == [0, 1, 2]


def test_run_batched_inductive(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy, transductive=False, num_pretranslations=3, batch_size=2)
    env.job.run()

    pretranslations = json.loads(env.target_pretranslations)
    assert len(pretranslations) == 3
    for pretranslation in pretranslations:
        assert len(pretranslation["alignment"]) > 0
    assert env.align_batch_sizes == [2, 1]


def test_cancel(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    checker = _CancellationChecker(3)
    with raises(CanceledError):
        env.job.run(check_canceled=checker.check_canceled)

    assert env.target_pretranslations == ""


def test_post_edit_before_alignment(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy, transductive=False)
    env.config.update({"post_edit_model": "openai/example"})
    decoy.when(env.translation_file_service.create_source_corpus()).then_return(
        DictionaryTextCorpus(MemoryText("text", [TextRow("text", 1, ["Por", "favor"])]))
    )
    decoy.when(env.translation_file_service.create_target_corpus()).then_return(
        DictionaryTextCorpus(MemoryText("text", [TextRow("text", 1, ["Please"])]))
    )
    client = MockLlmCompletionClient([Completion('{"translation":"Please, I booked a room."}')])
    stages = []
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
        patch(
            "machine.translation.llm.openrouter_llm_completion_client.OpenRouterLlmCompletionClient",
            return_value=client,
        ) as factory,
    ):
        env.job.run(progress=lambda status: stages.append(getattr(status, "phase_stage", None)))

    result = json.loads(env.target_pretranslations)[0]
    assert result["translation"] == "Please, I have booked a room."
    assert result["postEditedTranslation"] == "Please, I booked a room."
    assert result["translationTokens"] == ["Please", ",", "I", "booked", "a", "room", "."]
    assert result["sequenceConfidence"] == 0.5
    assert result["sourceRefs"] == ["ref1"]
    assert "Please</target>" in client.requests[0]["messages"][1]["content"]
    assert stages.index("inference") < stages.index("post_edit") < stages.index("align")
    factory.assert_called_once_with(model="openai/example", api_key="test")


def test_post_edit_refusal_preserves_draft_without_alignment(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    env.config.update({"post_edit_model": "openai/example", "align_pretranslations": False})
    client = MockLlmCompletionClient([RefusalError([], "refused")])
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
        patch(
            "machine.translation.llm.openrouter_llm_completion_client.OpenRouterLlmCompletionClient",
            return_value=client,
        ),
    ):
        env.job.run()

    result = json.loads(env.target_pretranslations)[0]
    assert result["translation"] == "Please, I have booked a room."
    assert result["sequenceConfidence"] == 0.5


def test_post_edit_refusal_preserves_draft_whitespace(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy, draft=" draft  ")
    env.config.update({"post_edit_model": "openai/example", "align_pretranslations": False})
    client = MockLlmCompletionClient([RefusalError([], "refused")])
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
        patch(
            "machine.translation.llm.openrouter_llm_completion_client.OpenRouterLlmCompletionClient",
            return_value=client,
        ),
    ):
        env.job.run()

    assert json.loads(env.target_pretranslations)[0]["translation"] == " draft  "


def test_post_edit_requires_api_key(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    env.config.update({"post_edit_model": "openai/example"})
    with patch.dict(os.environ, {}, clear=True), raises(RuntimeError, match="OPENROUTER_API_KEY"):
        env.job.run()


def test_post_edit_cancellation(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    env.config.update({"post_edit_model": "openai/example"})
    client = MockLlmCompletionClient([Completion('{"translation":"Changed."}')])

    def progress(status) -> None:
        if status.phase_stage == "post_edit" and status.phase_step == 1:
            raise CanceledError()

    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
        patch(
            "machine.translation.llm.openrouter_llm_completion_client.OpenRouterLlmCompletionClient",
            return_value=client,
        ),
        raises(CanceledError),
    ):
        env.job.run(progress=progress)


def test_remote_build_post_edits_drafts(decoy: Decoy, caplog) -> None:
    env = _TestEnvironment(decoy, transductive=False)
    env.config.update(
        {
            "post_edit_model": "openai/editor",
            "align_pretranslations": False,
            "remote_llm": {"num_examples": 0, "batch_size": 1, "num_workers": 1},
        }
    )
    drafting_client = MockLlmCompletionClient(
        [
            ContentBlockError([], "blocked", cost=0.2, input_token_count=100, output_token_count=5),
            Completion("A draft.", 0.1, input_token_count=50, output_token_count=3),
        ]
    )
    postediting_client = MockLlmCompletionClient([Completion('{"translation":"An edited draft."}', 0.5)])
    job = RemoteLlmEngineBuildJob(
        env.config,
        RemoteLlmModelFactory(env.config, drafting_client, ExampleRetrievalMethod.TFIDF),
        env.translation_file_service,
        env.word_alignment_model_factory,
        drafting_client,
    )
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
        patch(
            "machine.translation.llm.openrouter_llm_completion_client.OpenRouterLlmCompletionClient",
            return_value=postediting_client,
        ),
        patch("machine.translation.llm.remote_llm_engine.sleep"),
        caplog.at_level("INFO"),
    ):
        job.run()

    result = json.loads(env.target_pretranslations)[0]
    assert result["translation"] == "A draft."
    assert result["postEditedTranslation"] == "An edited draft."
    assert "A draft." in postediting_client.requests[0]["messages"][1]["content"]
    assert "Remote LLM usage: 2 requests, 150 prompt tokens, 8 completion tokens, $0.300000" in caplog.text
    assert postediting_client.total_cost == 0.5


def test_post_edit_empty_pretranslations(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy, num_pretranslations=0)
    env.config.update({"post_edit_model": "openai/example", "align_pretranslations": False})
    client = MockLlmCompletionClient([])
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
        patch(
            "machine.translation.llm.openrouter_llm_completion_client.OpenRouterLlmCompletionClient",
            return_value=client,
        ),
    ):
        env.job.run()

    assert json.loads(env.target_pretranslations) == []
    assert client.requests == []


class _TransductiveModel(WordAlignmentModel, TransductiveWordAlignmentModel):
    pass


def _create_translation_result(translation: str = "Please, I have booked a room.") -> TranslationResult:
    return TranslationResult(
        translation=translation,
        source_tokens="Por favor , tengo reservada una habitación .".split(),
        target_tokens="Please , I have booked a room .".split(),
        confidences=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        sequence_confidence=0.5,
        sources=[TranslationSources.NMT] * 8,
        alignment=WordAlignmentMatrix.from_word_pairs(
            8, 8, {(0, 0), (1, 0), (2, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)}
        ),
        phrases=[Phrase(Range.create(0, 8), 8)],
    )


class _TestEnvironment:
    def __init__(
        self,
        decoy: Decoy,
        transductive: bool = False,
        num_pretranslations: int = 1,
        batch_size: int = 100,
        source_text: str = "Por favor, tengo reservada una habitación.",
        draft: str = "Please, I have booked a room.",
    ) -> None:
        self.source_tokenizer_trainer = decoy.mock(cls=Trainer)
        self.target_tokenizer_trainer = decoy.mock(cls=Trainer)
        decoy.when(self.source_tokenizer_trainer.__enter__()).then_return(self.source_tokenizer_trainer)
        decoy.when(self.target_tokenizer_trainer.__enter__()).then_return(self.target_tokenizer_trainer)

        self.model_trainer = decoy.mock(cls=Trainer)
        decoy.when(self.model_trainer.__enter__()).then_return(self.model_trainer)
        stats = TrainStats()
        stats.train_corpus_size = 3
        stats.metrics["bleu"] = 30.0
        decoy.when(self.model_trainer.stats).then_return(stats)

        self.translate_batch_sizes: List[int] = []

        def _translate_batch(segments: List[str]) -> List[TranslationResult]:
            self.translate_batch_sizes.append(len(segments))
            return [_create_translation_result(draft) for _ in segments]

        self.engine = decoy.mock(cls=TranslationEngine)
        decoy.when(self.engine.__enter__()).then_return(self.engine)
        decoy.when(self.engine.translate_batch(matchers.Anything())).then_do(_translate_batch)

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
                            textId=f"text{i + 1}",
                            sourceRefs=[f"ref{i + 1}"],
                            targetRefs=[f"ref{i + 1}"],
                            translation=source_text,
                            sourceTokens=[],
                            translationTokens=[],
                            alignment="",
                            sequenceConfidence=0.5,
                        )
                        for i in range(num_pretranslations)
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

        alignment = WordAlignmentMatrix.from_word_pairs(
            row_count=8,
            column_count=8,
            set_values=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        )
        self.training_alignment_requests: List[int] = []
        self.align_batch_sizes: List[int] = []

        def _get_training_alignment(n: int) -> WordAlignmentMatrix:
            self.training_alignment_requests.append(n)
            return alignment

        def _align_batch(segments: List[object]) -> List[WordAlignmentMatrix]:
            self.align_batch_sizes.append(len(segments))
            return [alignment for _ in segments]

        if transductive:
            self.model = decoy.mock(cls=_TransductiveModel)
            decoy.when(self.model.__enter__()).then_return(self.model)
            decoy.when(self.model.get_training_alignment(matchers.Anything())).then_do(_get_training_alignment)
        else:
            self.model = decoy.mock(cls=WordAlignmentModel)
            decoy.when(self.model.__enter__()).then_return(self.model)
            decoy.when(self.model.align_batch(matchers.Anything())).then_do(_align_batch)

        self.word_alignment_model_factory = decoy.mock(cls=WordAlignmentModelFactory)
        decoy.when(
            self.word_alignment_model_factory.create_model_trainer(matchers.Anything(), matchers.Anything())
        ).then_return(self.alignment_model_trainer)
        decoy.when(self.word_alignment_model_factory.create_alignment_model()).then_return(self.model)
        decoy.when(self.word_alignment_model_factory.save_model()).then_return(Path("model.zip"))

        self.config = MockSettings(
            {
                "src_lang": "es",
                "trg_lang": "en",
                "save_model": "save-model",
                "inference_batch_size": batch_size,
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
        )
        self.job = HuggingFaceNmtEngineBuildJob(
            self.config,
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

from types import SimpleNamespace

from machine.corpora.parallel_text_corpus import ParallelTextCorpus
from machine.corpora.parallel_text_row import ParallelTextRow
from machine.jobs.remote_llm_engine_build_job import RemoteLlmEngineBuildJob
from machine.jobs.thot.thot_word_alignment_model_factory import ThotWordAlignmentModelFactory
from machine.jobs.translation_file_service import PretranslationInfo
from machine.translation.llm.example import Example
from machine.translation.llm.example_retriever_factory import ExampleRetrieverFactory
from machine.translation.llm.example_retriever_method import ExampleRetrievalMethod
from machine.translation.llm.llm_completion_client import LlmCompletionClient
from machine.translation.llm.openrouter_llm_completion_client import RefusalError
from machine.translation.llm.remote_llm_engine import RemoteLlmEngine
from machine.utils.phased_progress_reporter import PhasedProgressReporter


def test_build_job_uses_only_nonempty_segment_pairs_as_examples() -> None:
    rows = [
        ParallelTextRow("text", [1], [1], ["hello"], ["hola"]),
        ParallelTextRow("text", [3], [3], [], ["empty source"]),
    ]
    parallel_corpus = ParallelTextCorpus.from_parallel_rows(rows)
    job = RemoteLlmEngineBuildJob(
        SimpleNamespace(src_lang="English", trg_lang="Spanish"),
        completion_client=None,  # type: ignore[arg-type]
        translation_file_service=None,  # type: ignore[arg-type]
        word_alignment_model_factory=ThotWordAlignmentModelFactory(SimpleNamespace()),
        retrieval_method=ExampleRetrievalMethod.TFIDF,
    )

    example_count, confidence = job._train_model(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        parallel_corpus,
        PhasedProgressReporter(None, []),
        None,
    )

    assert example_count == 1
    assert confidence != confidence
    assert job._example_retriever is not None
    assert job._example_retriever.examples == [Example("hello", "hola")]


class _AlwaysRefusingClient(LlmCompletionClient):
    def __init__(self) -> None:
        self.request_count = 0

    def complete(self, messages, logprobs=False, response_schema=None):
        self.request_count += 1
        raise RefusalError(messages)


def test_build_job_tracks_verse_refs_when_translation_retries_are_exhausted() -> None:
    config = {"src_lang": "eng", "trg_lang": "spa", "inference_batch_size": 4}
    client = _AlwaysRefusingClient()
    job = RemoteLlmEngineBuildJob(
        config,
        client,
        None,  # type: ignore[arg-type]
        ThotWordAlignmentModelFactory(config),
    )
    engine = RemoteLlmEngine(
        client,
        ExampleRetrieverFactory.create_example_retriever(ExampleRetrievalMethod.TFIDF, []),
        "English",
        "Spanish",
        num_workers=1,
    )
    pretranslation: PretranslationInfo = {
        "corpusId": "corpus",
        "textId": "GEN",
        "sourceRefs": ["GEN 1:1"],
        "targetRefs": ["GEN 1:1"],
        "translation": "In the beginning",
        "sourceTokens": [],
        "translationTokens": [],
        "alignment": "",
        "sequenceConfidence": 0,
    }

    translated = list(job._translate(engine, [pretranslation], 1, lambda _: None, None))

    assert translated[0][1]["translation"] == ""
    assert client.request_count == 3
    assert [(verse.text_id, verse.source_refs) for verse in job.untranslated_verses] == [("GEN", ("GEN 1:1",))]

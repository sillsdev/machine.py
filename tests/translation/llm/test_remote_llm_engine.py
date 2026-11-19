import re
from math import exp
from threading import Lock
from time import sleep

from machine.translation.llm.example import Example
from machine.translation.llm.example_retriever import ExampleRetriever
from machine.translation.llm.llm_completion_client import Completion, LlmCompletionClient, LlmUsageTotals, TokenLogprob
from machine.translation.llm.openrouter_llm_completion_client import ContentBlockError, RefusalError
from machine.translation.llm.remote_llm_engine import RemoteLlmEngine
from machine.translation.translation_sources import TranslationSources


class _Retriever(ExampleRetriever):
    def __init__(self, examples):
        self.queries = []
        super().__init__(examples)

    def _fit(self) -> None:
        pass

    def _rank(self, query: str, k: int, exclude_index=None):
        self.queries.append(query)
        return self._examples[:k]


class _Client(LlmCompletionClient):
    def __init__(self) -> None:
        self.messages = []
        self.logprobs_requested = False

    def complete(self, messages, logprobs=False):
        self.messages = messages
        self.logprobs_requested = logprobs
        return Completion("Hello, world!", 0.01, [TokenLogprob(" Hello", -0.2), TokenLogprob("!", -0.4)])

    def supports_logprobs(self) -> bool:
        return True


class _BatchClient(LlmCompletionClient):
    def __init__(self, replies, delay=0):
        self.replies = iter(replies)
        self.calls = []
        self.delay = delay
        self.lock = Lock()

    def complete(self, messages, logprobs=False):
        if self.delay:
            sleep(self.delay)
        with self.lock:
            self.calls.append(messages)
            reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _EchoBatchClient(LlmCompletionClient):
    def __init__(self):
        self.calls = []
        self.lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def complete(self, messages, logprobs=False):
        content = messages[-1]["content"]
        sources = re.findall(r"^\d+\. source (\d+)$", content, flags=re.MULTILINE)
        with self.lock:
            self.calls.append(messages)
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        sleep(0.02)
        with self.lock:
            self.active_calls -= 1
        if sources:
            reply = "\n".join(f"{i}. target {source}" for i, source in enumerate(sources, 1))
        else:
            source = re.search(r"source (\d+)$", content)
            reply = "" if source is None else f"target {source.group(1)}"
        return Completion(reply, 0.1, input_token_count=100, output_token_count=20)


def test_remote_llm_engine_builds_icl_prompt_and_result() -> None:
    client = _Client()
    examples = [Example("buenos días", "good morning"), Example("gracias", "thank you")]
    retriever = _Retriever(examples)
    engine = RemoteLlmEngine(client, retriever, "spa", "eng", num_examples=1)

    result = engine.translate("hola mundo")

    assert client.messages == [
        {
            "role": "system",
            "content": (
                "Translate from spa to eng. Use the examples as guidance. "
                "Return only the translation, with no explanation."
            ),
        },
        {"role": "user", "content": "buenos días"},
        {"role": "assistant", "content": "good morning"},
        {"role": "user", "content": "hola mundo"},
    ]
    assert client.logprobs_requested
    assert retriever.queries == ["hola mundo"]
    assert result.translation == "Hello, world!"
    assert result.source_tokens == ["hola", "mundo"]
    assert result.target_tokens == ["Hello", ",", "world", "!"]
    assert result.sources == [TranslationSources.LLM] * 4
    assert result.confidences == [-1.0] * 4
    assert result.sequence_confidence == exp(-0.3)
    assert result.alignment.row_count == 2
    assert result.alignment.column_count == 4


def test_remote_llm_engine_batches_at_fifteen_and_preserves_input_order():
    sources = [f"source {i}" for i in range(31)]
    client = _EchoBatchClient()
    engine = RemoteLlmEngine(client, _Retriever([]), "Source", "Target", num_workers=3)

    results = engine.translate_batch(sources)

    assert [result.translation for result in results] == [f"target {i}" for i in range(31)]
    assert len(client.calls) == 3
    batch_sizes = [
        len(re.findall(r"^\d+\. source", call[-1]["content"], flags=re.MULTILINE)) or 1 for call in client.calls
    ]
    assert sorted(batch_sizes) == [1, 15, 15]
    assert client.max_active_calls > 1
    assert engine.usage_totals.summary.request_count == 3
    assert engine.usage_totals.summary.input_token_count == 300
    assert engine.usage_totals.summary.output_token_count == 60
    assert round(engine.usage_totals.summary.reported_cost, 6) == 0.3
    assert engine.usage_totals.summary.unpriced_request_count == 0


def test_remote_llm_engine_corrects_then_splits_a_malformed_batch():
    client = _BatchClient(
        [
            Completion("not numbered", 0.01),
            Completion("still malformed", 0.02),
            Completion("uno", 0.03),
            Completion("dos", 0.04),
        ]
    )
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng", batch_size=2)

    results = engine.translate_batch(["uno", "dos"])

    assert [result.translation for result in results] == ["uno", "dos"]
    assert len(client.calls) == 4
    assert engine.usage_totals.summary.request_count == 4
    assert engine.usage_totals.summary.reported_cost == 0.1


def test_remote_llm_engine_retries_refusal_then_splits_and_tracks_failed_segments():
    client = _BatchClient(
        [
            ContentBlockError([]),
            ContentBlockError([]),
            ContentBlockError([]),
            Completion("hola"),
            RefusalError([]),
            RefusalError([]),
            RefusalError([]),
        ]
    )
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng", batch_size=2, num_workers=1)

    results = engine.translate_batch(["source one", "source two"])

    assert [result.translation for result in results] == ["hola", ""]
    assert len(client.calls) == 7
    assert engine.failed_segment_indices == [1]


def test_remote_llm_engine_skips_empty_sources_and_retries_empty_single_reply():
    client = _BatchClient([Completion("```text\n```", 0.01), Completion("Translation: hola", 0.02)])
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng")

    results = engine.translate_batch(["", "hola"])

    assert [result.translation for result in results] == ["", "hola"]
    assert len(client.calls) == 2
    assert engine.usage_totals.summary.request_count == 2
    assert engine.usage_totals.summary.reported_cost == 0.03


def test_usage_totals_sum_only_reported_token_counts():
    totals = LlmUsageTotals()
    totals.add(Completion("hola", cost=None, input_token_count=None, output_token_count=3))

    summary = totals.summary

    assert summary.request_count == 1
    assert summary.input_token_count == 0
    assert summary.output_token_count == 3
    assert summary.reported_cost == 0
    assert summary.unpriced_request_count == 1


def test_remote_llm_engine_translate_n_batch_accepts_a_scalar_string():
    client = _Client()
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng")

    results = engine.translate_n_batch(1, "hola mundo")

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].source_tokens == ["hola", "mundo"]

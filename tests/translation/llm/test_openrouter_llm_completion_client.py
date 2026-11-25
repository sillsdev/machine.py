from contextlib import contextmanager
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

import httpx
import pytest
from openrouter import errors

from machine.translation.llm import (
    CompletionFinishReason,
    ContentBlockError,
    MalformedCompletionError,
    OpenRouterLlmCompletionClient,
    RefusalError,
    RemoteLlmEngine,
    TfidfExampleRetriever,
    TruncatedCompletionError,
)
from machine.translation.llm.llm_completion_client import Completion, TokenLogprob


class _FakeChat:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, list):
            response = self.response[len(self.calls) - 1]
            if isinstance(response, Exception):
                raise response
            return response
        return self.response


class _FakeModels:
    def __init__(self, supported_parameters, error=None):
        self.supported_parameters = supported_parameters
        self.error = error
        self.calls = []

    def get(self, *, author, slug):
        self.calls.append((author, slug))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=SimpleNamespace(supported_parameters=self.supported_parameters))


class _FakeOpenRouter:
    def __init__(self, response, supported_parameters=None, model_error=None):
        self.chat = _FakeChat(response)
        self.models = _FakeModels(supported_parameters or [], model_error)
        self.enter_calls = 0
        self.exit_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_calls += 1


@contextmanager
def _client(response, supported_parameters=None, model_error=None):
    sdk_client = _FakeOpenRouter(response, supported_parameters, model_error)
    with patch("openrouter.OpenRouter", return_value=sdk_client):
        yield sdk_client


def test_openrouter_client_sends_chat_request_and_maps_completion_fields():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hola"))],
        usage=SimpleNamespace(cost=0.012, prompt_tokens=8, completion_tokens=2),
    )
    with _client(response) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", max_tokens=50, api_key="test-key") as completion_client:
            result = completion_client.complete([{"role": "user", "content": "hello"}])

    assert result == Completion("hola", 0.012, input_token_count=8, output_token_count=2)
    assert sdk_client.chat.calls == [
        {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 50,
            "logprobs": False,
            "stream": False,
        }
    ]
    assert sdk_client.enter_calls == 1
    assert sdk_client.exit_calls == 1


def test_openrouter_client_max_tokens_is_none():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))], usage=None)
    with _client(response) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            client.complete([])

    assert sdk_client.chat.calls == [
        {
            "model": "openai/gpt-4o",
            "messages": [],
            "logprobs": False,
            "stream": False,
        }
    ]


def test_openrouter_client_retry_budget():
    with patch("openrouter.OpenRouter") as sdk:
        OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key")

    assert sdk.call_args.kwargs["retry_config"].backoff.max_elapsed_time == 600000


def test_openrouter_client_merges_default_and_per_call_kwargs_without_changing_model():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hola"))],
        usage=None,
    )
    with _client(response) as sdk_client:
        with OpenRouterLlmCompletionClient(
            "openai/gpt-4o", api_key="test-key", seed=1, provider={"order": ["openai"]}
        ) as client:
            client.complete([], seed=42, reasoning_effort="high", model="other/model")

    assert sdk_client.chat.calls[0] == {
        "model": "openai/gpt-4o",
        "messages": [],
        "logprobs": False,
        "stream": False,
        "seed": 42,
        "provider": {"order": ["openai"]},
        "reasoning_effort": "high",
    }


def test_openrouter_client_rejects_response_without_choices():
    response = SimpleNamespace(choices=[], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(MalformedCompletionError, match="No choices"):
                client.complete([])


def test_openrouter_client_rejects_null_choice():
    response = SimpleNamespace(choices=[None], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(MalformedCompletionError, match="first choice is null"):
                client.complete([])


def test_openrouter_client_rejects_null_message():
    response = SimpleNamespace(choices=[SimpleNamespace(message=None)], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(MalformedCompletionError, match="no message"):
                client.complete([])


def test_openrouter_client_rejects_response_with_null_content():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(MalformedCompletionError, match="no content"):
                client.complete([])


def test_openrouter_client_rejects_response_with_non_text_content():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=42))], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(MalformedCompletionError, match="content is not text"):
                client.complete([])


def test_openrouter_client_fails_after_client_is_closed():
    response = SimpleNamespace(choices=[], usage=None)
    with _client(response):
        client = OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key")
        with client:
            pass

        with pytest.raises(RuntimeError, match="closed"):
            client.complete([])

        with pytest.raises(RuntimeError, match="closed"):
            with client:
                client.complete([])


def test_openrouter_client_maps_requested_token_logprobs():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hola"),
                logprobs=SimpleNamespace(content=[SimpleNamespace(token="hola", logprob=-0.2)]),
            )
        ],
        usage=None,
    )
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            result = client.complete([], logprobs=True)

    assert result == Completion("hola", None, [TokenLogprob("hola", -0.2)])


def test_openrouter_client_reports_logprobs_supported_by_its_model_and_caches_lookup():
    response = SimpleNamespace(choices=[], usage=None)
    with _client(response, ["temperature", "logprobs"]) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert client.supports_logprobs()
            assert client.supports_logprobs()

    assert sdk_client.models.calls == [("openai", "gpt-4o")]


def test_openrouter_client_reports_logprobs_unsupported_by_its_model():
    response = SimpleNamespace(choices=[], usage=None)
    with _client(response, ["temperature"]):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert not client.supports_logprobs()


def test_openrouter_client_reports_logprobs_unsupported_when_model_lookup_fails():
    response = SimpleNamespace(choices=[], usage=None)
    with _client(response, model_error=RuntimeError("metadata unavailable")):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert not client.supports_logprobs()


def test_openrouter_client_reports_logprobs_unsupported_for_unqualified_model_ids():
    response = SimpleNamespace(choices=[], usage=None)
    with _client(response, ["logprobs"]) as sdk_client:
        with OpenRouterLlmCompletionClient("gpt-4o", api_key="test-key") as client:
            assert not client.supports_logprobs()
            assert sdk_client.models.calls == []


def test_openrouter_client_raises_content_block_error():
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="content_filter")])
    with _client(response):
        with OpenRouterLlmCompletionClient("gpt-4o", api_key="test-key") as client:
            messages = [{"role": "user", "content": "hello"}]
            with pytest.raises(ContentBlockError, check=lambda e: e.messages == messages):
                client.complete(messages)


@pytest.mark.parametrize("content", [None, ""])
def test_openrouter_client_raises_refusal_error(content):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content, refusal="I can't help"), finish_reason="stop")
        ]
    )
    with _client(response):
        with OpenRouterLlmCompletionClient("gpt-4o", api_key="test-key") as client:
            messages = [{"role": "user", "content": "hello"}]
            with pytest.raises(RefusalError, match="I can't help", check=lambda e: e.messages == messages):
                client.complete(messages)


def test_complete_response_schema():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))], usage=None)
    with _client(response) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            client.complete([], response_schema={"type": "object", "properties": {}})

    assert sdk_client.chat.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


@pytest.mark.parametrize(
    "reported,expected",
    [
        ("stop", CompletionFinishReason.STOP),
        ("length", CompletionFinishReason.LENGTH),
        ("error", CompletionFinishReason.UNKNOWN),
        (None, CompletionFinishReason.UNKNOWN),
    ],
)
def test_complete_finish_reason(reported: Optional[str], expected: CompletionFinishReason):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hola"), finish_reason=reported)],
        usage=None,
    )
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert client.complete([]).finish_reason is expected


def _schema_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(cost=0.5, prompt_tokens=10, completion_tokens=2),
    )


_SCHEMA = {"type": "object", "properties": {"translation": {"type": "string"}}, "required": ["translation"]}


@pytest.mark.parametrize(
    "content,match",
    [
        ("Sure! Here you go", "not JSON"),
        ('{"translation": null}', "does not match"),
        ("{}", "does not match"),
        ('{"translation": "hola", "extra": 1}', "does not match"),
        ('["hola"]', "does not match"),
    ],
)
def test_complete_rejects_response_not_matching_schema(content: str, match: str):
    with _client(_schema_response(content)):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(MalformedCompletionError, match=match) as error:
                client.complete([], response_schema=_SCHEMA)

    assert (error.value.cost, error.value.input_token_count, error.value.output_token_count) == (0.5, 10, 2)


def test_complete_accepts_response_matching_schema():
    with _client(_schema_response('{"translation": "hola"}')):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            result = client.complete([], response_schema=_SCHEMA)

    assert result.text == '{"translation": "hola"}'


def test_complete_does_not_validate_without_schema():
    with _client(_schema_response("Sure! Here you go")):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert client.complete([]).text == "Sure! Here you go"


@pytest.mark.parametrize(
    "choice,expected_error",
    [
        (SimpleNamespace(finish_reason="content_filter"), ContentBlockError),
        (SimpleNamespace(message=SimpleNamespace(content=None, refusal="I can't help")), RefusalError),
        (None, MalformedCompletionError),
    ],
)
def test_complete_errors_carry_billed_usage(choice, expected_error):
    response = SimpleNamespace(choices=[choice], usage=SimpleNamespace(cost=0.5, prompt_tokens=10, completion_tokens=2))
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(expected_error) as error:
                client.complete([])

    assert (error.value.cost, error.value.input_token_count, error.value.output_token_count) == (0.5, 10, 2)


@pytest.mark.parametrize("content", ['{"translation": "ho', None, ""])
def test_complete_raises_truncated_error_with_schema(content: Optional[str]):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="length")],
        usage=SimpleNamespace(cost=0.5, prompt_tokens=10, completion_tokens=2),
    )
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", max_tokens=2, api_key="test-key") as client:
            with pytest.raises(TruncatedCompletionError) as error:
                client.complete([], response_schema=_SCHEMA)

    assert (error.value.cost, error.value.input_token_count, error.value.output_token_count) == (0.5, 10, 2)


def test_complete_returns_truncated_reply_without_schema():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hol"), finish_reason="length")], usage=None
    )
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", max_tokens=2, api_key="test-key") as client:
            result = client.complete([])

    assert result.text == "hol"
    assert result.finish_reason is CompletionFinishReason.LENGTH


def test_complete_running_totals_include_failed_responses():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Sure!"), finish_reason="stop")],
        usage=SimpleNamespace(cost=0.5, prompt_tokens=10, completion_tokens=2),
    )
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert (client.total_cost, client.total_input_token_count, client.total_output_token_count) == (0.0, 0, 0)
            client.complete([])
            with pytest.raises(MalformedCompletionError):
                client.complete([], response_schema=_SCHEMA)

    assert (client.total_cost, client.total_input_token_count, client.total_output_token_count) == (1.0, 20, 4)


def test_retry_usage_includes_billed_blocked_response():
    blocked = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="content_filter")],
        usage=SimpleNamespace(cost=0.2, prompt_tokens=100, completion_tokens=5),
    )
    success = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hola"), finish_reason="stop")],
        usage=SimpleNamespace(cost=0.1, prompt_tokens=50, completion_tokens=3),
    )
    with _client([blocked, success]), patch("machine.translation.llm.remote_llm_engine.sleep"):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            engine = RemoteLlmEngine(client, TfidfExampleRetriever([]), "English", "Spanish")
            assert engine.translate("hello").translation == "hola"

    summary = client.usage_summary
    assert summary.request_count == 2
    assert summary.reported_cost == pytest.approx(0.3)
    assert (summary.input_token_count, summary.output_token_count) == (150, 8)
    assert summary.unpriced_request_count == 0


def test_usage_summary_counts_missing_price():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hola"))], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            completion = client.complete([])

    assert completion.cost is None
    assert client.usage_summary.request_count == 1
    assert client.usage_summary.unpriced_request_count == 1
    assert client.usage_summary.reported_cost == 0


@contextmanager
def _sdk_client(responses: List[httpx.Response]):
    """Runs the real SDK, including its retries, against canned HTTP responses and records each retry's sleep."""
    import openrouter

    requests: List[httpx.Request] = []
    sleeps: List[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses[len(requests) - 1]

    real_openrouter = openrouter.OpenRouter

    def create(**kwargs):
        return real_openrouter(client=httpx.Client(transport=httpx.MockTransport(handle)), **kwargs)

    with (
        patch("openrouter.OpenRouter", side_effect=create),
        patch("openrouter.utils.retries.time.sleep", side_effect=sleeps.append),
    ):
        yield requests, sleeps


_OK = httpx.Response(
    200,
    json={
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "system_fingerprint": None,
        "model": "openai/gpt-4o",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hola"}}],
    },
)


def _error(status_code: int, headers=None) -> httpx.Response:
    return httpx.Response(status_code, headers=headers, json={"error": {"code": status_code, "message": "failed"}})


def test_complete_retries_rate_limit_after_retry_after():
    with _sdk_client([_error(429, {"Retry-After": "3"}), _OK]) as (requests, sleeps):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert client.complete([]).text == "hola"

    assert len(requests) == 2
    assert sleeps == [3.0]


def test_complete_retries_server_errors():
    with _sdk_client([_error(503), _OK]) as (requests, sleeps):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert client.complete([]).text == "hola"

    assert len(requests) == 2
    assert len(sleeps) == 1


def test_complete_does_not_retry_other_http_errors():
    with _sdk_client([_error(400), _OK]) as (requests, sleeps):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(errors.OpenRouterError):
                client.complete([])

    assert len(requests) == 1
    assert sleeps == []

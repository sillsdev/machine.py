import subprocess
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

import httpx
import pytest
from openrouter import errors
from openrouter.components import TooManyRequestsResponseErrorData

from machine.translation.llm import (
    CompletionFinishReason,
    ContentBlockError,
    MalformedCompletionError,
    OpenRouterLlmCompletionClient,
    RefusalError,
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

    assert result == Completion("hola", 0.0, [TokenLogprob("hola", -0.2)])


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


def _rate_limit_error(headers=None):
    raw_response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://openrouter.ai"))
    data = errors.TooManyRequestsResponseErrorData(
        error=TooManyRequestsResponseErrorData(code=429, message="slow down")
    )
    return errors.TooManyRequestsResponseError(data, raw_response)


_HOLA = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hola"))], usage=None)


class _InstantEvent(threading.Event):
    """Returns from each wait at once, recording its timeout, so that backoff is checked without waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: List[Optional[float]] = []

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.timeouts.append(timeout)
        return self.is_set()


def test_complete_retries_rate_limit_with_backoff():
    stop_event = _InstantEvent()
    with _client([_rate_limit_error(), _rate_limit_error(), _HOLA]) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key", stop_event=stop_event) as client:
            assert client.complete([]).text == "hola"

    assert len(sdk_client.chat.calls) == 3
    first, second = stop_event.timeouts
    assert first is not None and 0.5 <= first <= 1.0
    assert second is not None and 1.0 <= second <= 2.0


@pytest.mark.parametrize("retry_after,expected", [("3", 3.0), ("600", 60.0), ("-1", 0.0)])
def test_complete_rate_limit_honors_retry_after(retry_after: str, expected: float):
    stop_event = _InstantEvent()
    with _client([_rate_limit_error({"Retry-After": retry_after}), _HOLA]):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key", stop_event=stop_event) as client:
            client.complete([])

    assert stop_event.timeouts == [expected]


@pytest.mark.parametrize("retry_after", ["nan", "inf", "-inf"])
def test_complete_rate_limit_backs_off_for_non_finite_retry_after(retry_after: str):
    stop_event = _InstantEvent()
    with _client([_rate_limit_error({"Retry-After": retry_after}), _HOLA]):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key", stop_event=stop_event) as client:
            assert client.complete([]).text == "hola"

    (timeout,) = stop_event.timeouts
    assert timeout is not None and 0.5 <= timeout <= 1.0


def test_complete_rate_limit_gives_up_after_max_retries():
    stop_event = _InstantEvent()
    with _client([_rate_limit_error(), _rate_limit_error(), _rate_limit_error()]) as sdk_client:
        with OpenRouterLlmCompletionClient(
            "openai/gpt-4o", api_key="test-key", max_rate_limit_retries=2, stop_event=stop_event
        ) as client:
            with pytest.raises(errors.TooManyRequestsResponseError):
                client.complete([])

    assert len(sdk_client.chat.calls) == 3
    assert len(stop_event.timeouts) == 2


def test_complete_rate_limit_stops_when_stop_event_is_set():
    stop_event = threading.Event()
    stop_event.set()
    with _client([_rate_limit_error(), _HOLA]) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key", stop_event=stop_event) as client:
            with pytest.raises(errors.TooManyRequestsResponseError):
                client.complete([])

    assert len(sdk_client.chat.calls) == 1


def test_complete_does_not_retry_other_http_errors():
    stop_event = _InstantEvent()
    server_error = errors.OpenRouterError(
        "bad request", httpx.Response(400, request=httpx.Request("POST", "https://openrouter.ai"))
    )
    with _client([server_error, _HOLA]) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key", stop_event=stop_event) as client:
            with pytest.raises(errors.OpenRouterError, match="bad request"):
                client.complete([])

    assert len(sdk_client.chat.calls) == 1
    assert stop_event.timeouts == []


def test_client_rejects_negative_max_rate_limit_retries():
    with _client(_HOLA):
        with pytest.raises(ValueError, match="max_rate_limit_retries"):
            OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key", max_rate_limit_retries=-1)


def test_package_imports_without_jsonschema():
    # A fresh interpreter, so that the package's modules are imported with jsonschema blocked.
    code = "import sys; sys.modules['jsonschema'] = None; from machine.translation.llm import TfidfExampleRetriever"

    subprocess.run([sys.executable, "-c", code], check=True)


def test_complete_with_schema_requires_jsonschema_before_sending(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with _client(_HOLA) as sdk_client:
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            assert client.complete([]).text == "hola"
            with pytest.raises(ImportError, match="jsonschema"):
                client.complete([], response_schema={"type": "object"})

    assert len(sdk_client.chat.calls) == 1

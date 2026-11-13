from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from machine.translation.llm import ContentBlockError, OpenRouterLlmCompletionClient
from machine.translation.llm.llm_completion_client import Completion, TokenLogprob


class _FakeChat:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
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
            with pytest.raises(ValueError, match="No choices"):
                client.complete([])


def test_openrouter_client_rejects_null_choice():
    response = SimpleNamespace(choices=[None], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(ValueError, match="first choice is null"):
                client.complete([])


def test_openrouter_client_rejects_null_message():
    response = SimpleNamespace(choices=[SimpleNamespace(message=None)], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(ValueError, match="no message"):
                client.complete([])


def test_openrouter_client_rejects_response_with_null_content():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(ValueError, match="no content"):
                client.complete([])


def test_openrouter_client_rejects_response_with_non_text_content():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=42))], usage=None)
    with _client(response):
        with OpenRouterLlmCompletionClient("openai/gpt-4o", api_key="test-key") as client:
            with pytest.raises(ValueError, match="content is not text"):
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

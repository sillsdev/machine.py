import logging
from typing import Any, Dict, List, Optional

from .llm_completion_client import Completion, LlmCompletionClient, TokenLogprob

LOGGER = logging.getLogger(__name__)


class OpenRouterLlmCompletionClient(LlmCompletionClient):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._logprobs_supported: Optional[bool] = None
        self._entered = False
        self._closed = False
        try:
            from openrouter import OpenRouter
        except ImportError as error:
            raise ImportError(
                "OpenRouter completions require the 'openrouter' package. Install it with `uv sync --extra llm`."
            ) from error
        self._client = OpenRouter(api_key=api_key)

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        request = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "logprobs": logprobs,
            "stream": False,
        }
        if self._max_tokens is not None:
            request["max_completion_tokens"] = self._max_tokens
        response = self._client.chat.send(**request)
        choices = getattr(response, "choices", [])
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        text = content if isinstance(content, str) else ""
        token_logprobs = self._get_logprobs(choice) if logprobs else []
        usage = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        return Completion(
            text=text,
            cost=float(cost) if isinstance(cost, (int, float)) else 0.0,
            token_logprobs=token_logprobs,
            input_token_count=int(prompt_tokens) if isinstance(prompt_tokens, int) else 0,
            output_token_count=int(completion_tokens) if isinstance(completion_tokens, int) else 0,
        )

    def supports_logprobs(self) -> bool:
        if self._logprobs_supported is None:
            try:
                author, separator, slug = self._model.partition("/")
                if not separator or not author or not slug:
                    return False
                model = self._client.models.get(author=author, slug=slug).data
                self._logprobs_supported = "logprobs" in model.supported_parameters
            except Exception:
                LOGGER.warning("Could not determine whether OpenRouter model '%s' supports logprobs.", self._model)
                self._logprobs_supported = False
        return self._logprobs_supported

    def __enter__(self) -> "OpenRouterLlmCompletionClient":
        if not self._entered and not self._closed:
            self._client.__enter__()
            self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._entered:
            self._client.__exit__(exc_type, exc_value, traceback)
            self._entered = False
            self._closed = True

    def close(self) -> None:
        if not self._entered and not self._closed:
            self._client.__exit__(None, None, None)
            self._closed = True

    def _get_logprobs(self, choice: Any) -> List[TokenLogprob]:
        logprobs = getattr(choice, "logprobs", None)
        entries = getattr(logprobs, "content", None) or []
        return [TokenLogprob(str(entry.token), float(entry.logprob)) for entry in entries]

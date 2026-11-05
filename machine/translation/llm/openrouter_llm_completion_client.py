import logging
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from .llm_completion_client import Completion, LlmCompletionClient, TokenLogprob

logger = logging.getLogger(__name__)


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
        self._closed = False
        try:
            from openrouter import OpenRouter
        except ImportError as error:
            raise ImportError(
                "OpenRouter completions require the 'openrouter' package. Install it with `uv sync --extra llm`."
            ) from error
        self._client = OpenRouter(api_key=api_key)

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        if self._closed:
            raise RuntimeError("OpenRouter completion client is closed.")
        try:
            from openrouter.components import (
                ChatAssistantMessage,
                ChatAssistantMessageContent,
                ChatContentText,
                ChatResult,
                ChatUsage,
            )
        except ImportError as error:
            raise ImportError(
                "OpenRouter completions require the 'openrouter' package. Install it with `uv sync --extra llm`."
            ) from error
        request = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "logprobs": logprobs,
            "stream": False,
        }
        if self._max_tokens is not None:
            request["max_completion_tokens"] = self._max_tokens
        response: ChatResult = self._client.chat.send(**request)
        choice = response.choices[0] if response.choices else None
        if choice is not None and getattr(choice, "finish_reason", None) == "length" and self._max_tokens is not None:
            logger.warning("Completion was finished because the max tokens ceiling %d was hit", self._max_tokens)
        message: Optional[ChatAssistantMessage] = getattr(choice, "message", None)
        content: Optional[ChatAssistantMessageContent] = getattr(message, "content", None)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(p.text for p in content if isinstance(p, ChatContentText))
        else:
            text = ""
        token_logprobs = self._get_logprobs(choice) if logprobs else []
        usage: Optional[ChatUsage] = getattr(response, "usage", None)
        cost: Optional[float] = getattr(usage, "cost", None)
        prompt_tokens: int = getattr(usage, "prompt_tokens", 0)
        completion_tokens: int = getattr(usage, "completion_tokens", 0)
        return Completion(
            text=text,
            cost=cost or 0.0,
            token_logprobs=token_logprobs,
            input_token_count=prompt_tokens,
            output_token_count=completion_tokens,
        )

    def supports_logprobs(self) -> bool:
        if self._closed:
            raise RuntimeError("OpenRouter completion client is closed.")
        if self._logprobs_supported is None:
            try:
                author, separator, slug = self._model.partition("/")
                if not separator or not author or not slug:
                    return False
                model = self._client.models.get(author=author, slug=slug).data
                self._logprobs_supported = "logprobs" in model.supported_parameters
            except Exception:
                logger.warning(
                    "Could not determine whether OpenRouter model '%s' supports logprobs.", self._model, exc_info=True
                )
        return self._logprobs_supported or False

    def __enter__(self) -> "OpenRouterLlmCompletionClient":
        if self._closed:
            raise RuntimeError("OpenRouter completion client is closed.")
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._client.__exit__(exc_type, exc_value, traceback)
        self._closed = True

    def _get_logprobs(self, choice: Any) -> List[TokenLogprob]:
        logprobs = getattr(choice, "logprobs", None)
        entries = getattr(logprobs, "content", None) or []
        return [TokenLogprob(str(entry.token), float(entry.logprob)) for entry in entries]

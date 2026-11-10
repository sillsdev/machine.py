import logging
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from .llm_completion_client import Completion, LlmCompletionClient, TokenLogprob

logger = logging.getLogger(__name__)


class OpenRouterLlmCompletionClient(LlmCompletionClient):
    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **default_kwargs: Any,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._default_kwargs = default_kwargs
        self._logprobs_supported: Optional[bool] = None
        self._closed = False
        try:
            from openrouter import OpenRouter
        except ImportError as error:
            raise ImportError(
                "OpenRouter completions require the 'openrouter' package. Install it with `uv sync --extra llm`."
            ) from error
        self._client = OpenRouter(api_key=api_key)

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False, **kwargs: Any) -> Completion:
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
        request = dict(self._default_kwargs)
        request.update(kwargs)
        request.update(
            {
                "model": self._model,
                "messages": messages,
                "logprobs": logprobs,
                "stream": False,
            }
        )
        if self._max_tokens is not None:
            request["max_completion_tokens"] = self._max_tokens
        response: ChatResult = self._client.chat.send(**request)
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Malformed OpenRouter completion response: No choices were returned.")
        choice = choices[0]
        if choice is None:
            raise ValueError("Malformed OpenRouter completion response: The first choice is null.")
        if getattr(choice, "finish_reason", None) == "length" and self._max_tokens is not None:
            logger.warning("Completion was finished because the max tokens ceiling %d was hit", self._max_tokens)
        message: Optional[ChatAssistantMessage] = getattr(choice, "message", None)
        if message is None:
            raise ValueError("Malformed OpenRouter completion response: The first choice has no message.")
        content: Optional[ChatAssistantMessageContent] = getattr(message, "content", None)
        if content is None:
            raise ValueError("Malformed OpenRouter completion response: The message has no content.")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = [p.text for p in content if isinstance(p, ChatContentText) and isinstance(p.text, str)]
            if not text_parts:
                raise ValueError("Malformed OpenRouter completion response: The message has no text content.")
            text = "".join(text_parts)
        else:
            raise ValueError("Malformed OpenRouter completion response: The message content is not text.")
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

import json
import logging
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from .completion_finish_reason import CompletionFinishReason
from .llm_completion_client import (
    Completion,
    ContentBlockError,
    LlmCompletionClient,
    MalformedCompletionError,
    RefusalError,
    TokenLogprob,
    TruncatedCompletionError,
)

logger = logging.getLogger(__name__)


_FINISH_REASONS: Dict[Optional[str], CompletionFinishReason] = {
    "stop": CompletionFinishReason.STOP,
    "length": CompletionFinishReason.LENGTH,
}


class OpenRouterLlmCompletionClient(LlmCompletionClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: Optional[int] = None,
        **default_kwargs: Any,
    ) -> None:
        super().__init__()
        self._model = model
        self._max_tokens = max_tokens
        self._default_kwargs = default_kwargs
        self._logprobs_supported: Optional[bool] = None
        self._closed = False
        try:
            from openrouter import OpenRouter, RetryConfig
            from openrouter.utils import BackoffStrategy
        except ImportError as error:
            raise ImportError(
                "OpenRouter completions require the 'openrouter' package. Install it with `uv sync --extra llm`."
            ) from error
        # Retry-After takes precedence over backoff natively, so no need to handle that here
        self._client = OpenRouter(
            api_key=api_key,
            retry_config=RetryConfig(
                strategy="backoff",
                backoff=BackoffStrategy(500, 60000, 1.5, 600000),
                status_codes_override=["5XX", "429"],
                retry_connection_errors=True,
            ),
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        logprobs: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Completion:
        if self._closed:
            raise RuntimeError("OpenRouter completion client is closed.")
        try:
            from openrouter.components import (
                ChatAssistantMessage,
                ChatAssistantMessageContent,
                ChatChoice,
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
        schema: Optional[Dict[str, Any]] = None
        jsonschema: Any = None
        if response_schema is not None:
            jsonschema = _import_jsonschema()
            schema = {**response_schema, "additionalProperties": False}
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": schema},
            }
        response: ChatResult = self._client.chat.send(**request)
        usage: Optional[ChatUsage] = getattr(response, "usage", None)
        cost: Optional[float] = getattr(usage, "cost", None)
        prompt_tokens: Optional[int] = getattr(usage, "prompt_tokens", None)
        completion_tokens: Optional[int] = getattr(usage, "completion_tokens", None)
        billed_cost = cost or 0.0
        billed_prompt_tokens = prompt_tokens or 0
        billed_completion_tokens = completion_tokens or 0
        self._record_usage(cost, prompt_tokens, completion_tokens)

        usage_fields = {
            "cost": billed_cost,
            "input_token_count": billed_prompt_tokens,
            "output_token_count": billed_completion_tokens,
        }

        def malformed(reason: str) -> MalformedCompletionError:
            return MalformedCompletionError(
                messages, f"Malformed OpenRouter completion response: {reason}", **usage_fields
            )

        choices = getattr(response, "choices", None)
        if not choices:
            raise malformed("No choices were returned.")
        choice: Optional[ChatChoice] = choices[0]
        if choice is None:
            raise malformed("The first choice is null.")
        reported_finish_reason = getattr(choice, "finish_reason", None)
        if reported_finish_reason == "content_filter":
            raise ContentBlockError(
                messages, "Completion response terminated because of content filtering.", **usage_fields
            )
        if reported_finish_reason == "length":
            if schema is not None:
                raise TruncatedCompletionError(
                    messages, "Completion response was cut off by the output token limit.", **usage_fields
                )
            if self._max_tokens is not None:
                logger.warning("Completion was finished because the max tokens ceiling %d was hit", self._max_tokens)
        message: Optional[ChatAssistantMessage] = getattr(choice, "message", None)
        if message is None:
            raise malformed("The first choice has no message.")
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RefusalError(messages, f"Completion response was a refusal from the model: {refusal}", **usage_fields)
        content: Optional[ChatAssistantMessageContent] = getattr(message, "content", None)
        if content is None:
            raise malformed("The message has no content.")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = [p.text for p in content if isinstance(p, ChatContentText) and isinstance(p.text, str)]
            if not text_parts:
                raise malformed("The message has no text content.")
            text = "".join(text_parts)
        else:
            raise malformed("The message content is not text.")
        if schema is not None:
            try:
                jsonschema.validate(json.loads(text), schema)
            except json.JSONDecodeError as error:
                raise malformed(f"The message is not JSON: {error}") from error
            except jsonschema.ValidationError as error:
                raise malformed(f"The message does not match the response schema: {error.message}") from error
        token_logprobs = self._get_logprobs(choice) if logprobs else []
        return Completion(
            text=text,
            cost=None if cost is None else float(cost),
            token_logprobs=token_logprobs,
            input_token_count=None if prompt_tokens is None else int(prompt_tokens),
            output_token_count=None if completion_tokens is None else int(completion_tokens),
            finish_reason=_FINISH_REASONS.get(getattr(choice, "finish_reason", None), CompletionFinishReason.UNKNOWN),
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


def _import_jsonschema() -> Any:
    try:
        import jsonschema
    except ImportError as error:
        raise ImportError(
            "Validating a response schema requires the 'jsonschema' package. Install it with `uv sync --extra llm`."
        ) from error
    return jsonschema

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from .completion_finish_reason import CompletionFinishReason


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    """A reply from the model, with whatever the provider reported alongside it."""

    text: str
    cost: Optional[float] = None
    token_logprobs: List[TokenLogprob] = field(default_factory=list)
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    finish_reason: CompletionFinishReason = CompletionFinishReason.UNKNOWN


@dataclass(frozen=True)
class LlmUsageSummary:
    request_count: int
    input_token_count: int
    output_token_count: int
    reported_cost: float
    unpriced_request_count: int


class CompletionError(Exception):
    def __init__(
        self,
        messages: List[Dict[str, str]],
        reason: str,
        *,
        cost: float = 0.0,
        input_token_count: int = 0,
        output_token_count: int = 0,
    ) -> None:
        super().__init__(reason)
        self.messages = messages
        self.cost = cost
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count


class ContentBlockError(CompletionError):
    pass


class RefusalError(CompletionError):
    pass


class MalformedCompletionError(CompletionError):
    pass


class TruncatedCompletionError(CompletionError):
    pass


class LlmCompletionClient(ABC):
    def __init__(self) -> None:
        self._usage_lock = threading.Lock()
        self._total_cost = 0.0
        self._total_input_token_count = 0
        self._total_output_token_count = 0
        self._request_count = 0
        self._unpriced_request_count = 0

    @property
    def usage_summary(self) -> LlmUsageSummary:
        with self._usage_lock:
            return LlmUsageSummary(
                self._request_count,
                self._total_input_token_count,
                self._total_output_token_count,
                self._total_cost,
                self._unpriced_request_count,
            )

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_token_count(self) -> int:
        return self._total_input_token_count

    @property
    def total_output_token_count(self) -> int:
        return self._total_output_token_count

    @abstractmethod
    def complete(
        self, messages: List[Dict[str, str]], logprobs: bool = False, response_schema: Optional[Dict[str, Any]] = None
    ) -> Completion:
        """Raises ContentBlockError or RefusalError when the provider declines the request, and
        MalformedCompletionError when the reply has no text.

        Given a response_schema, raises TruncatedCompletionError when the reply hit the output token limit, and
        MalformedCompletionError when it is not JSON matching the schema. Without one, a truncated reply is returned
        with finish_reason LENGTH.

        Every billed response, including one that then raises, is added to the client's running totals through
        _record_usage. The totals span all callers sharing the client.
        """

    def supports_logprobs(self) -> bool:
        return False

    def _record_usage(
        self, cost: Optional[float], input_token_count: Optional[int], output_token_count: Optional[int]
    ) -> None:
        with self._usage_lock:
            self._request_count += 1
            if cost is None:
                self._unpriced_request_count += 1
            else:
                self._total_cost += cost
            self._total_input_token_count += input_token_count or 0
            self._total_output_token_count += output_token_count or 0

    def count_tokens(self, text: str) -> Optional[int]:
        return None

    def __enter__(self) -> LlmCompletionClient:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass

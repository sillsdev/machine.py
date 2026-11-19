from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
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


class LlmUsageTotals:
    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._input_token_count = 0
        self._output_token_count = 0
        self._reported_cost = 0.0
        self._unpriced_request_count = 0

    def add(self, completion: Completion) -> None:
        with self._lock:
            self._request_count += 1
            if completion.input_token_count is not None:
                self._input_token_count += completion.input_token_count
            if completion.output_token_count is not None:
                self._output_token_count += completion.output_token_count
            if completion.cost is None:
                self._unpriced_request_count += 1
            else:
                self._reported_cost += completion.cost

    @property
    def summary(self) -> LlmUsageSummary:
        with self._lock:
            return LlmUsageSummary(
                self._request_count,
                self._input_token_count,
                self._output_token_count,
                self._reported_cost,
                self._unpriced_request_count,
            )


class LlmCompletionClient(ABC):
    @abstractmethod
    def complete(
        self, messages: List[Dict[str, str]], logprobs: bool = False, response_schema: Optional[Dict[str, Any]] = None
    ) -> Completion: ...

    def supports_logprobs(self) -> bool:
        return False

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

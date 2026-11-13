from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from .completion_finish_reason import CompletionFinishReason


class ContentBlockError(Exception):
    def __init__(self, messages: List[Dict[str, str]]) -> None:
        super().__init__("Completion response terminated because of content filtering.")
        self.messages = messages


class RefusalError(Exception):
    def __init__(self, messages: List[Dict[str, str]]) -> None:
        super().__init__("Completion response was a refusal from the model.")
        self.messages = messages


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    """A reply from the model, with whatever the provider reported alongside it."""

    text: str
    cost: float
    token_logprobs: List[TokenLogprob] = field(default_factory=list)
    input_token_count: int = 0
    output_token_count: int = 0
    finish_reason: CompletionFinishReason = CompletionFinishReason.UNKNOWN


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

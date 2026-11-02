from dataclasses import dataclass, field
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


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
    # True for a reply saved by an earlier run, whose cost that run already paid.
    reused: bool = False


class LlmCompletionClient(ABC):
    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion: ...

    def supports_logprobs(self) -> bool:
        return False

    def count_tokens(self, text: str) -> Optional[int]:
        return None

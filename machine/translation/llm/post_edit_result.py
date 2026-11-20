from dataclasses import dataclass
from typing import Optional

from .completion_finish_reason import CompletionFinishReason


@dataclass(frozen=True)
class PostEditResult:
    translation: str
    succeeded: bool = False
    changed: bool = False
    justification: str = ""
    total_cost: float = 0.0
    total_input_token_count: int = 0
    total_output_token_count: int = 0
    finish_reason: Optional[CompletionFinishReason] = None

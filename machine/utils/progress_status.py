from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProgressStatus:
    @classmethod
    def from_step(cls, current_step: int, step_count: int, message: Optional[str] = None) -> "ProgressStatus":
        return ProgressStatus(1.0 if step_count == 0 else (current_step / step_count), message)

    percent_completed: float
    message: Optional[str] = None

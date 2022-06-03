from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProgressStatus:
    @classmethod
    def from_step(cls, step: int, step_count: int, message: Optional[str] = None) -> ProgressStatus:
        return ProgressStatus(step, 1.0 if step_count == 0 else (step / step_count), message)

    step: int
    percent_completed: Optional[float] = None
    message: Optional[str] = None

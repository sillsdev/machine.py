from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from ..utils.progress_status import ProgressStatus


class TrainStats:
    def __init__(self) -> None:
        self.trained_segment_count: int = 0
        self._metrics: Dict[str, float] = {}

    @property
    def metrics(self) -> Dict[str, float]:
        return self._metrics


class Trainer(ABC):
    @abstractmethod
    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @property
    @abstractmethod
    def stats(self) -> TrainStats:
        ...

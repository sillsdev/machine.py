from typing import Callable, Optional

from ..utils.progress_status import ProgressStatus
from .trainer import Trainer, TrainStats


class NullTrainer(Trainer):
    def __init__(self) -> None:
        self._stats = TrainStats()

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        pass

    def save(self) -> None:
        pass

    def stats(self) -> TrainStats:
        return self._stats

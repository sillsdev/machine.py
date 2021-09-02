from typing import Callable, Optional

from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .trainer import Trainer, TrainStats


class SymmetrizedWordAlignmentModelTrainer(Trainer):
    def __init__(self, direct_trainer: Trainer, inverse_trainer: Trainer) -> None:
        self._direct_trainer = direct_trainer
        self._inverse_trainer = inverse_trainer

    @property
    def stats(self) -> TrainStats:
        return self._direct_trainer.stats

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        reporter = PhasedProgressReporter(
            progress, [Phase("Training direct alignment model"), Phase("Training inverse alignment model")]
        )

        with reporter.start_next_phase() as phase_progress:
            self._direct_trainer.train(phase_progress, check_canceled)
        if check_canceled is not None:
            check_canceled()
        with reporter.start_next_phase() as phase_progress:
            self._inverse_trainer.train(phase_progress, check_canceled)

    def save(self) -> None:
        self._direct_trainer.save()
        self._inverse_trainer.save()

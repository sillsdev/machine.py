from typing import Callable, Optional

from ...utils.phased_progress_reporter import Phase, PhasedProgressReporter, PhaseProgress
from ...utils.progress_status import ProgressStatus

_TRAIN_PHASES = [
    Phase("Training language model", 0.01),
    Phase("Training direct alignment model", 0.2),
    Phase("Generating best direct alignments", report_steps=False),
    Phase("Training inverse alignment model", 0.2),
    Phase("Generating best inverse alignments", report_steps=False),
    Phase("Merging alignments"),
    Phase("Generating phrase table"),
    Phase("Tuning language model"),
    Phase("Tuning translation model", 0.4, report_steps=False),
    Phase("Finalizing", 0.05, report_steps=False),
]


class ThotTrainProgressReporter(PhasedProgressReporter):
    def __init__(
        self, progress: Optional[Callable[[ProgressStatus], None]], check_canceled: Optional[Callable[[], None]]
    ) -> None:
        super().__init__(progress, _TRAIN_PHASES)
        self._check_canceled = check_canceled

    def check_canceled(self) -> None:
        if self._check_canceled is not None:
            self._check_canceled()

    def start_next_phase(self) -> PhaseProgress:
        self.check_canceled()

        return super().start_next_phase()

    def _report(self, value: ProgressStatus) -> None:
        self.check_canceled()

        return super()._report(value)

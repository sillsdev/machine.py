from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Callable, Iterable, Optional, Sequence, Type

from .progress_status import ProgressStatus


@dataclass(frozen=True)
class Phase:
    message: Optional[str] = None
    percentage: float = 0


class PhaseProgress(AbstractContextManager):
    def __init__(self, reporter: "PhasedProgressReporter", phase: Phase) -> None:
        self._reporter = reporter
        self._phase = phase
        self._reporter.report(ProgressStatus(0))
        self._percent_completed = 0.0

    @property
    def phase(self) -> Phase:
        return self._phase

    def _report(self, value: ProgressStatus) -> None:
        self._percent_completed = value.percent_completed
        self._reporter.report(value)

    def __enter__(self) -> Callable[[ProgressStatus], None]:
        return self._report

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        if self._percent_completed < 1.0:
            self._reporter.report(ProgressStatus(1.0))


class PhasedProgressReporter:
    def __init__(self, progress: Optional[Callable[[ProgressStatus], None]], phases: Iterable[Phase]) -> None:
        self._progress = progress
        self._phases = list(phases)

        sum = 0
        unspecified_count = 0
        for phase in self._phases:
            sum += phase.percentage
            if phase.percentage == 0:
                unspecified_count += 1

        self._default_percentage = 0 if unspecified_count == 0 else ((1.0 - sum) / unspecified_count)
        self._current_phase_index = -1
        self._percent_completed = 0.0

    @property
    def phases(self) -> Sequence[Phase]:
        return self._phases

    @property
    def current_phase(self) -> Optional[Phase]:
        return None if self._current_phase_index == -1 else self._phases[self._current_phase_index]

    def start_next_phase(self) -> PhaseProgress:
        self._percent_completed += self._current_phase_percentage
        self._current_phase_index += 1

        return PhaseProgress(self, self._phases[self._current_phase_index])

    def report(self, value: ProgressStatus) -> None:
        if self._progress is None:
            return

        percent_completed = self._percent_completed + (self._current_phase_percentage * value.percent_completed)
        message = self._phases[self._current_phase_index].message if value.message is None else value.message
        self._progress(ProgressStatus(percent_completed, message))

    @property
    def _current_phase_percentage(self) -> float:
        if self.current_phase is None:
            return 0
        pcnt = self.current_phase.percentage
        if pcnt == 0:
            pcnt = self._default_percentage
        return pcnt

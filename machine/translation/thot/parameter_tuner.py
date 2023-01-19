from abc import ABC, abstractmethod
from typing import Sequence

from ...utils import ProgressStatus
from ..trainer import TrainStats
from .thot_smt_parameters import ThotSmtParameters


class ParameterTuner(ABC):
    @abstractmethod
    def tune(
        self,
        parameters: ThotSmtParameters,
        tune_source_corpus: Sequence[Sequence[str]],
        tune_target_corpus: Sequence[Sequence[str]],
        stats: TrainStats,
        progress: ProgressStatus,
    ) -> ThotSmtParameters:
        ...

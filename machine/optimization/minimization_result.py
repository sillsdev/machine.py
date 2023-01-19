from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class MinimizationExitCondition(Enum):
    NONE = auto()
    CONVERGED = auto()
    MAX_FUNCTION_EVALUATIONS = auto()


@dataclass
class MinimizationResult:
    reason_for_exit: MinimizationExitCondition
    minimizing_point: np.ndarray
    error_value: float
    function_evaluation_count: int

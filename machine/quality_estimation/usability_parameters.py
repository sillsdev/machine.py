from dataclasses import dataclass
from typing import ClassVar


@dataclass
class UsabilityParameters:
    unusable: ClassVar["UsabilityParameters"]
    usable: ClassVar["UsabilityParameters"]

    count: float
    mean: float
    variance: float


UsabilityParameters.unusable = UsabilityParameters(count=97.0, mean=45.85, variance=99.91)
UsabilityParameters.usable = UsabilityParameters(count=263.0, mean=51.4, variance=95.19)

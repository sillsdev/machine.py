from enum import Enum, auto


class SymmetrizationHeuristic(Enum):
    NONE = auto()
    UNION = auto()
    INTERSECTION = auto()
    OCH = auto()
    GROW = auto()
    GROW_DIAG = auto()
    GROW_DIAG_FINAL = auto()
    GROW_DIAG_FINAL_AND = auto()

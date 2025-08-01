from enum import Enum, auto


class QuotationMarkUpdateStrategy(Enum):
    APPLY_FULL = auto()
    APPLY_FALLBACK = auto()
    SKIP = auto()

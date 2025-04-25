from enum import Enum, auto


class QuotationDenormalizationAction(Enum):
    APPLY_FULL = auto()
    APPLY_BASIC = auto()
    SKIP = auto()

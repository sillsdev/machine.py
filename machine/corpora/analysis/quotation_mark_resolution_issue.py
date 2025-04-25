from enum import Enum, auto


class QuotationMarkResolutionIssue(Enum):
    UNPAIRED_QUOTATION_MARK = auto()
    TOO_DEEP_NESTING = auto()
    INCOMPATIBLE_QUOTATION_MARK = auto()
    AMBIGUOUS_QUOTATION_MARK = auto()
    UNEXPECTED_QUOTATION_MARK = auto()

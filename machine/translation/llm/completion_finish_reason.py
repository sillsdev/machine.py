from enum import Enum, auto


class CompletionFinishReason(Enum):
    STOP = auto()
    LENGTH = auto()
    CONTENT_FILTER = auto()
    REFUSAL = auto()
    UNKNOWN = auto()

from enum import Enum, auto


class EditOperation(Enum):
    NONE = auto()
    HIT = auto()
    INSERT = auto()
    DELETE = auto()
    PREFIX_DELETE = auto()
    SUBSTITUTE = auto()

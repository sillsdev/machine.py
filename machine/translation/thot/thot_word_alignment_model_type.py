from enum import IntEnum, auto


class ThotWordAlignmentModelType(IntEnum):
    FAST_ALIGN = auto()
    IBM1 = auto()
    IBM2 = auto()
    HMM = auto()
    IBM3 = auto()
    IBM4 = auto()


def getThotWordAlignmentModelType(str) -> ThotWordAlignmentModelType:
    return ThotWordAlignmentModelType.__dict__[str.upper()]


def checkThotWordAlignmentModelType(str) -> bool:
    return str.upper() in ThotWordAlignmentModelType.__dict__

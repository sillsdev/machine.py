from enum import Enum, auto


class UpdateUsfmTextBehavior(Enum):
    PREFER_EXISTING = auto()
    PREFER_NEW = auto()
    STRIP_EXISTING = auto()


class UpdateUsfmMarkerBehavior(Enum):
    PRESERVE = auto()
    STRIP = auto()

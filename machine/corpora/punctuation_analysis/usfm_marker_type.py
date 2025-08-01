from enum import Enum, auto


class UsfmMarkerType(Enum):
    PARAGRAPH = auto()
    CHARACTER = auto()
    VERSE = auto()
    CHAPTER = auto()
    EMBED = auto()
    OTHER = auto()
    NO_MARKER = auto()

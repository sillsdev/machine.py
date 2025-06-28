from enum import Enum, auto


class UsfmMarkerType(Enum):
    ParagraphMarker = auto()
    CharacterMarker = auto()
    VerseMarker = auto()
    ChapterMarker = auto()
    EmbedMarker = auto()
    Other = auto()
    NoMarker = auto()

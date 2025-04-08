from enum import Enum


class UsfmMarkerType(Enum):
    ParagraphMarker = "ParagraphMarker"
    CharacterMarker = "CharacterMarker"
    VerseMarker = "VerseMarker"
    ChapterMarker = "ChapterMarker"
    EmbedMarker = "Embed"
    Other = "Other"
    NoMarker = "NoMarker"

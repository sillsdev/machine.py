from enum import Enum, Flag, auto
from typing import Optional, Set


class UsfmTextType(Enum):
    TITLE = auto()
    SECTION = auto()
    VERSE_TEXT = auto()
    NOTE_TEXT = auto()
    OTHER = auto()
    BACK_TRANSLATION = auto()
    TRANSLATION_NOTE = auto()


class UsfmJustification(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()
    BOTH = auto()


class UsfmStyleType(Enum):
    UNKNOWN = auto()
    CHARACTER = auto()
    NOTE = auto()
    PARAGRAPH = auto()
    END = auto()


class UsfmTextProperties(Flag):
    NONE = 0
    VERSE = auto()
    CHAPTER = auto()
    PARAGRAPH = auto()
    PUBLISHABLE = auto()
    VERNACULAR = auto()
    POETIC = auto()
    OTHER_TEXT_BEGIN = auto()
    OTHER_TEXT_END = auto()
    LEVEL1 = auto()
    LEVEL2 = auto()
    LEVEL3 = auto()
    LEVEL4 = auto()
    LEVEL5 = auto()
    CROSS_REFERENCE = auto()
    NONPUBLISHABLE = auto()
    NONVERNACULAR = auto()
    BOOK = auto()
    NOTE = auto()


class UsfmMarker:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.bold: bool = False
        self.description: Optional[str] = None
        self.encoding: Optional[str] = None
        self.end_marker: Optional[str] = None
        self.first_line_indent: int = 0
        self.font_name: Optional[str] = None
        self.font_size: int = 0
        self.italic: bool = False
        self.justification: UsfmJustification = UsfmJustification.LEFT
        self.left_margin: int = 0
        self.line_spacing: int = 0
        self.name: Optional[str] = None
        self.not_repeatable: bool = False
        self._occurs_under: Set[str] = set()
        self.rank: int = 0
        self.right_margin: int = 0
        self.small_caps: bool = False
        self.space_after: int = 0
        self.space_before: int = 0
        self.style_type: UsfmStyleType = UsfmStyleType.UNKNOWN
        self.subscript: bool = False
        self.superscript: bool = False
        self.text_properties: UsfmTextProperties = UsfmTextProperties.NONE
        self.text_type: UsfmTextType = UsfmTextType.TITLE
        self.underline: bool = False
        self.xml_tag: Optional[str] = None
        self.regular: bool = False
        self.color: int = 0

    @property
    def occurs_under(self) -> Set[str]:
        return self._occurs_under

    def __repr__(self) -> str:
        return f"\\{self.marker}"

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, List, Optional, Sequence

from ..scripture.verse_ref import VerseRef, Versification
from .usfm_stylesheet import UsfmStylesheet
from .usfm_tag import UsfmTag, UsfmTextType
from .usfm_token import UsfmAttribute, UsfmToken


class UsfmElementType(Enum):
    BOOK = auto()
    PARA = auto()
    CHAR = auto()
    TABLE = auto()
    ROW = auto()
    CELL = auto()
    NOTE = auto()
    SIDEBAR = auto()


@dataclass
class UsfmParserElement:
    type: UsfmElementType
    marker: Optional[str]
    attributes: Optional[Sequence[UsfmAttribute]] = None


class UsfmParserState:
    def __init__(self, stylesheet: UsfmStylesheet, versification: Versification, tokens: Sequence[UsfmToken]) -> None:
        self._stylesheet = stylesheet
        self._stack: List[UsfmParserElement] = []
        self.verse_ref = VerseRef(versification=versification)
        self.verse_offset = 0
        self._tokens = tokens
        self.index = -1
        self.special_token = False

    @property
    def stylesheet(self) -> UsfmStylesheet:
        return self._stylesheet

    @property
    def tokens(self) -> Sequence[UsfmToken]:
        return self._tokens

    @property
    def token(self) -> Optional[UsfmToken]:
        return self._tokens[self.index] if self.index >= 0 else None

    @property
    def prev_token(self) -> Optional[UsfmToken]:
        return self._tokens[self.index - 1] if self.index >= 1 else None

    @property
    def stack(self) -> Sequence[UsfmParserElement]:
        return self._stack

    @property
    def is_figure(self) -> bool:
        return self.char_tag is not None and self.char_tag.marker == "fig"

    @property
    def para_tag(self) -> Optional[UsfmTag]:
        elem = next(
            (
                e
                for e in reversed(self._stack)
                if e.type in {UsfmElementType.PARA, UsfmElementType.BOOK, UsfmElementType.ROW, UsfmElementType.SIDEBAR}
            ),
            None,
        )
        if elem is not None:
            assert elem.marker is not None
            return self._stylesheet.get_tag(elem.marker)
        return None

    @property
    def char_tag(self) -> Optional[UsfmTag]:
        return next(iter(self.char_tags), None)

    @property
    def char_tags(self) -> Iterable[UsfmTag]:
        return (
            self._stylesheet.get_tag(e.marker)
            for e in reversed(self._stack)
            if e.type == UsfmElementType.CHAR and e.marker is not None
        )

    @property
    def note_tag(self) -> Optional[UsfmTag]:
        elem = next((e for e in reversed(self._stack) if e.type == UsfmElementType.NOTE), None)
        return self._stylesheet.get_tag(elem.marker) if elem is not None and elem.marker is not None else None

    @property
    def is_verse_para(self) -> bool:
        # If the user enters no markers except just \c and \v we want the text to be considered verse text. This is
        # covered by the empty stack that makes para_tag=None. Not specified text type is verse text
        para_tag = self.para_tag
        return (
            para_tag is None
            or para_tag.text_type == UsfmTextType.VERSE_TEXT
            or para_tag.text_type == UsfmTextType.NOT_SPECIFIED
        )

    @property
    def is_verse_text(self) -> bool:
        # Sidebars and notes are not verse text
        if any(e.type in {UsfmElementType.SIDEBAR, UsfmElementType.NOTE} for e in self._stack):
            return False

        if not self.is_verse_para:
            return False

        # All character tags must be verse text
        for char_tag in self.char_tags:
            # Not specified text type is verse text
            if char_tag.text_type != UsfmTextType.VERSE_TEXT and char_tag.text_type != UsfmTextType.NOT_SPECIFIED:
                return False

        return True

    @property
    def is_special_text(self) -> bool:
        return self.special_token

    def peek(self) -> UsfmParserElement:
        return self._stack[-1]

    def push(self, elem: UsfmParserElement) -> None:
        self._stack.append(elem)

    def pop(self) -> UsfmParserElement:
        return self._stack.pop()

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .usfm_marker import UsfmMarker


class UsfmTokenType(Enum):
    BOOK = auto()
    CHAPTER = auto()
    VERSE = auto()
    TEXT = auto()
    PARAGRAPH = auto()
    CHARACTER = auto()
    NOTE = auto()
    END = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class UsfmToken:
    type: UsfmTokenType
    marker: Optional[UsfmMarker]
    text: Optional[str]
    is_nested: bool = False

    def __repr__(self) -> str:
        string = ""
        if self.marker is not None:
            string += f"\\+{self.marker.marker}" if self.is_nested else str(self.marker)
        if self.text is not None and self.text != "":
            if len(string) > 0:
                string += " "
            string += self.text
        return string

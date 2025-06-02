from dataclasses import dataclass
from enum import Enum, auto

from .usfm_token import UsfmToken


class UsfmUpdateBlockElementType(Enum):
    TEXT = auto()
    PARAGRAPH = auto()
    EMBED = auto()
    STYLE = auto()
    OTHER = auto()


@dataclass
class UsfmUpdateBlockElement:
    type: UsfmUpdateBlockElementType
    tokens: list[UsfmToken]
    marked_for_removal: bool = False

    def get_tokens(self) -> list[UsfmToken]:
        if self.marked_for_removal:
            return []
        return self.tokens.copy()

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .usfm_token import UsfmToken


class ScriptureUpdateElementType(Enum):
    EXISTING_TEXT = auto()
    INSERTED_TEXT = auto()
    OTHER = auto()


@dataclass
class ScriptureUpdateElement:
    type: ScriptureUpdateElementType
    tokens: list[UsfmToken]
    marked_for_removal: bool = False

    def get_tokens(self) -> list[UsfmToken]:
        if self.marked_for_removal:
            return []
        return self.tokens

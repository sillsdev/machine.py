from dataclasses import dataclass
from enum import Enum, auto

from .update_usfm_behavior import UpdateUsfmMarkerBehavior
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

    def get_text(self) -> str:
        return "".join(t.to_usfm() for t in self.tokens)

    def is_placeable(
        self, paragraph_behavior: UpdateUsfmMarkerBehavior, style_behavior: UpdateUsfmMarkerBehavior
    ) -> bool:
        if self.marked_for_removal:
            return False
        if self.type == UsfmUpdateBlockElementType.PARAGRAPH:
            return paragraph_behavior == UpdateUsfmMarkerBehavior.PRESERVE and len(self.tokens) == 1
        if self.type == UsfmUpdateBlockElementType.STYLE:
            return style_behavior == UpdateUsfmMarkerBehavior.PRESERVE
        return False

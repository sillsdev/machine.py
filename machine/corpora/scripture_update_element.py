from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .scripture_embed import is_embed_style
from .usfm_token import UsfmToken, UsfmTokenType


class ScriptureUpdateElementType(Enum):
    EXISTING_TEXT = auto()
    INSERTED_TEXT = auto()
    PARAGRAPH = auto()
    EMBED = auto()
    STYLE = auto()
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


def create_non_text_scripture_element(
    tokens: list[UsfmToken], marked_for_removal: bool = False
) -> ScriptureUpdateElement:
    tokens = tokens.copy()
    # Determine if it is a Paragraph, style, embed or other
    if len(tokens) == 0 or tokens[0].marker is None:
        return ScriptureUpdateElement(ScriptureUpdateElementType.OTHER, [], marked_for_removal)
    if tokens[0].type == UsfmTokenType.PARAGRAPH:
        return ScriptureUpdateElement(ScriptureUpdateElementType.PARAGRAPH, tokens, marked_for_removal)
    if is_embed_style(tokens[0].marker):
        return ScriptureUpdateElement(ScriptureUpdateElementType.EMBED, tokens, marked_for_removal)
    else:
        return ScriptureUpdateElement(ScriptureUpdateElementType.STYLE, tokens, marked_for_removal)

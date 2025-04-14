from __future__ import annotations

from .scripture_ref import ScriptureRef
from .scripture_update_element import (
    ScriptureUpdateElement,
    ScriptureUpdateElementType,
    create_non_text_scripture_element,
)
from .usfm_token import UsfmToken, UsfmTokenType


class ScriptureUpdateBlock:

    def __init__(self) -> None:
        self.ref: ScriptureRef = ScriptureRef()
        self._elements: list[ScriptureUpdateElement] = []

    @property
    def elements(self) -> list[ScriptureUpdateElement]:
        return self._elements

    def add_existing_text(self, token: UsfmToken, marked_for_removal: bool = False) -> None:
        self._elements.append(
            ScriptureUpdateElement(ScriptureUpdateElementType.EXISTING_TEXT, [token], marked_for_removal)
        )

    def add_inserted_text(self, tokens: list[UsfmToken]) -> None:
        self._elements.append(ScriptureUpdateElement(ScriptureUpdateElementType.INSERTED_TEXT, tokens.copy()))

    def add_token(self, token: UsfmToken, marked_for_removal: bool = False) -> None:
        if token.type == UsfmTokenType.TEXT:
            self._elements.append(
                ScriptureUpdateElement(ScriptureUpdateElementType.EXISTING_TEXT, [token], marked_for_removal)
            )
        else:
            self._elements.append(create_non_text_scripture_element([token], marked_for_removal))

    def add_embed(self, tokens: list[UsfmToken], marked_for_removal: bool = False) -> None:
        if len(tokens) == 0:
            return
        self._elements.append(
            ScriptureUpdateElement(ScriptureUpdateElementType.EMBED_BLOCK, tokens, marked_for_removal)
        )

    def update_ref(self, ref: ScriptureRef) -> None:
        self.ref = ref

    def clear(self) -> None:
        self._elements.clear()
        self.ref = ScriptureRef()

    def get_tokens(self) -> list[UsfmToken]:
        return [token for element in self._elements for token in element.get_tokens()]

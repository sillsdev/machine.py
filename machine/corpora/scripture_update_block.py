from __future__ import annotations

from .scripture_ref import ScriptureRef
from .scripture_update_element import ScriptureUpdateElement, ScriptureUpdateElementType
from .usfm_token import UsfmToken, UsfmTokenType


class ScriptureUpdateBlock:

    def __init__(self) -> None:
        self._ref: ScriptureRef = ScriptureRef()
        self._elements: list[ScriptureUpdateElement] = []

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
            self._elements.append(ScriptureUpdateElement(ScriptureUpdateElementType.OTHER, [token], marked_for_removal))

    def add_tokens(self, tokens: list[UsfmToken], marked_for_removal: bool = False) -> None:
        if len(tokens) == 0:
            return
        self._elements.append(
            ScriptureUpdateElement(ScriptureUpdateElementType.OTHER, tokens.copy(), marked_for_removal)
        )

    def update_ref(self, ref: ScriptureRef) -> None:
        self._ref = ref

    def clear(self) -> None:
        self._elements.clear()
        self._ref = ScriptureRef()

    def get_tokens(self) -> list[UsfmToken]:
        return [token for element in self._elements for token in element.get_tokens()]

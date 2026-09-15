from __future__ import annotations

from typing import Iterable, Sequence

from .scripture_ref import ScriptureRef
from .usfm_token import UsfmToken, UsfmTokenType
from .usfm_update_block_element import UsfmUpdateBlockElement, UsfmUpdateBlockElementType


class UsfmUpdateBlock:
    def __init__(
        self,
        refs: Iterable[ScriptureRef] = [],
        elements: Iterable[UsfmUpdateBlockElement] = [],
        row_metadata: Iterable[dict[str, object]] = [],
    ) -> None:
        self._refs: list[ScriptureRef] = list(refs)
        self._elements: list[UsfmUpdateBlockElement] = list(elements)
        # One entry per row matched to this block, in order. A verse range can be matched by
        # several rows, in which case this block's text is those rows' texts concatenated.
        self._row_metadata: list[dict[str, object]] = list(row_metadata)

    @property
    def refs(self) -> Sequence[ScriptureRef]:
        return self._refs

    @property
    def elements(self) -> Sequence[UsfmUpdateBlockElement]:
        return self._elements

    @property
    def metadata(self) -> dict[str, object]:
        return self._row_metadata[-1] if len(self._row_metadata) > 0 else {}

    @property
    def row_metadata(self) -> Sequence[dict[str, object]]:
        return self._row_metadata

    def add_text(self, tokens: Iterable[UsfmToken]) -> None:
        self._elements.append(UsfmUpdateBlockElement(UsfmUpdateBlockElementType.TEXT, list(tokens)))

    def add_token(self, token: UsfmToken, marked_for_removal: bool = False) -> None:
        if token.type == UsfmTokenType.TEXT:
            element_type = UsfmUpdateBlockElementType.TEXT
        elif token.type == UsfmTokenType.PARAGRAPH:
            element_type = UsfmUpdateBlockElementType.PARAGRAPH
        elif token.type == UsfmTokenType.CHARACTER or token.type == UsfmTokenType.END:
            element_type = UsfmUpdateBlockElementType.STYLE
        else:
            element_type = UsfmUpdateBlockElementType.OTHER
        self._elements.append(UsfmUpdateBlockElement(element_type, [token], marked_for_removal))

    def add_embed(self, tokens: Iterable[UsfmToken], marked_for_removal: bool = False) -> None:
        self._elements.append(
            UsfmUpdateBlockElement(UsfmUpdateBlockElementType.EMBED, list(tokens), marked_for_removal)
        )

    def extend_last_element(self, tokens: Iterable[UsfmToken]) -> None:
        self._elements[-1].tokens.extend(tokens)

    def update_refs(self, refs: Iterable[ScriptureRef]) -> None:
        self._refs = list(refs)

    def get_last_paragraph(self) -> UsfmUpdateBlockElement | None:
        for element in reversed(self._elements):
            if element.type == UsfmUpdateBlockElementType.PARAGRAPH:
                return element
        return None

    def pop(self) -> UsfmUpdateBlockElement:
        return self._elements.pop()

    def get_tokens(self) -> list[UsfmToken]:
        return [token for element in self._elements for token in element.get_tokens()]

    def __eq__(self, other: UsfmUpdateBlock) -> bool:
        return (
            self._refs == other._refs
            and self._elements == other._elements
            and self._row_metadata == other._row_metadata
        )

    def copy(self) -> UsfmUpdateBlock:
        return UsfmUpdateBlock(self._refs, self._elements, self._row_metadata)

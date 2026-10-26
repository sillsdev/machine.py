from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .scripture_ref import ScriptureRef
from .usfm_token import UsfmToken
from .usfm_update_block_element import UsfmUpdateBlockElement


@dataclass
class UsfmUpdateBlockRow:
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


class UsfmUpdateBlock:
    def __init__(
        self,
        refs: Iterable[ScriptureRef] = [],
        elements: Iterable[UsfmUpdateBlockElement] = [],
        *,
        rows: Iterable[UsfmUpdateBlockRow] = [],
    ) -> None:
        self._refs: list[ScriptureRef] = list(refs)
        self._elements: list[UsfmUpdateBlockElement] = list(elements)
        # Several rows can match a verse range, and this block's text is their texts concatenated in order
        self._rows: list[UsfmUpdateBlockRow] = list(rows)

    @property
    def refs(self) -> Sequence[ScriptureRef]:
        return self._refs

    @property
    def elements(self) -> Sequence[UsfmUpdateBlockElement]:
        return self._elements

    @property
    def rows(self) -> Sequence[UsfmUpdateBlockRow]:
        return self._rows

    def get_tokens(self) -> list[UsfmToken]:
        return [token for element in self._elements for token in element.get_tokens()]

    def __eq__(self, other: UsfmUpdateBlock) -> bool:
        return self._refs == other._refs and self._elements == other._elements and self._rows == other._rows

    def copy(self) -> UsfmUpdateBlock:
        return UsfmUpdateBlock(self._refs, self._elements, rows=self._rows)

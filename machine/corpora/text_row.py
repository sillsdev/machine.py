from enum import Flag, auto
from typing import Any, Sequence


class TextRowFlags(Flag):
    NONE = 0
    SENTENCE_START = auto()
    IN_RANGE = auto()
    RANGE_START = auto()


class TextRow(Sequence[str]):
    def __init__(
        self, text_id: str, ref: Any, segment: Sequence[str] = [], flags: TextRowFlags = TextRowFlags.SENTENCE_START
    ) -> None:
        self._text_id = text_id
        self._ref = ref

        self.segment = segment
        self.flags = flags

    @property
    def text_id(self) -> str:
        return self._text_id

    @property
    def ref(self) -> Any:
        return self._ref

    @property
    def text(self) -> str:
        return " ".join(self.segment)

    @property
    def is_sentence_start(self) -> bool:
        return TextRowFlags.SENTENCE_START in self.flags

    @property
    def is_in_range(self) -> bool:
        return TextRowFlags.IN_RANGE in self.flags

    @property
    def is_range_start(self) -> bool:
        return TextRowFlags.RANGE_START in self.flags

    @property
    def is_empty(self) -> bool:
        return len(self.segment) == 0

    def __len__(self) -> int:
        return len(self.segment)

    def __getitem__(self, i: int) -> str:
        return self.segment[i]

    def __repr__(self) -> str:
        if self.is_empty:
            segment = "<range>" if self.is_in_range else "EMPTY"
        elif len(self.segment) > 0:
            segment = " ".join(self.segment)
        else:
            segment = "NONEMPTY"
        return f"{self.ref} - {segment}"

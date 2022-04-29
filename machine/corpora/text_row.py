from typing import Any, Sequence


class TextRow:
    def __init__(
        self,
        text_id: str,
        ref: Any,
        segment: Sequence[str] = [],
        is_sentence_start: bool = True,
        is_in_range: bool = False,
        is_range_start: bool = False,
        is_empty: bool = True,
    ) -> None:
        self._text_id = text_id
        self._ref = ref

        self.segment = segment
        self.is_sentence_start = is_sentence_start
        self.is_in_range = is_in_range
        self.is_range_start = is_range_start
        self.is_empty = is_empty

    @property
    def text_id(self) -> str:
        return self._text_id

    @property
    def ref(self) -> Any:
        return self._ref

    @property
    def text(self) -> str:
        return " ".join(self.segment)

    def __repr__(self) -> str:
        if self.is_empty:
            segment = "<range>" if self.is_in_range else "EMPTY"
        elif len(self.segment) > 0:
            segment = " ".join(self.segment)
        else:
            segment = "NONEMPTY"
        return f"{self.ref} - {segment}"

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class TextCorpusRow:
    ref: Any
    segment: Sequence[str]
    is_sentence_start: bool
    is_in_range: bool
    is_range_start: bool
    is_empty: bool

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

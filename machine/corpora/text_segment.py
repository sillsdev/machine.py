from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(eq=False, frozen=True)
class TextSegment:
    @classmethod
    def create(
        cls,
        text_id: str,
        seg_ref: Any,
        segment: Sequence[str],
        sentence_start: bool = True,
        in_range: bool = False,
        range_start: bool = False,
    ) -> "TextSegment":
        return TextSegment(text_id, seg_ref, segment, sentence_start, in_range, range_start, len(segment) == 0)

    @classmethod
    def create_empty(
        cls,
        text_id: str,
        seg_ref: Any,
        sentence_start: bool = True,
        in_range: bool = False,
        range_start: bool = False,
        empty: bool = True,
    ) -> "TextSegment":
        return TextSegment(text_id, seg_ref, [], sentence_start, in_range, range_start, empty)

    text_id: str
    segment_ref: Any
    segment: Sequence[str]
    sentence_start: bool
    is_in_range: bool
    is_range_start: bool
    is_empty: bool

    def __repr__(self) -> str:
        if self.is_empty:
            segment = "<range>" if self.is_in_range else "EMPTY"
        elif len(self.segment) > 0:
            segment = " ".join(self.segment)
        else:
            segment = "NONEMPTY"
        return f"{self.segment_ref} - {segment}"

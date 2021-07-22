from typing import Any, List, Optional
from dataclasses import dataclass


@dataclass(eq=False, frozen=True)
class TextSegment:
    segment_ref: Any
    segment: List[str]
    sentence_start: bool
    is_in_range: bool
    is_range_start: bool
    is_empty: bool

    def __init__(
        self,
        segment_ref: Any,
        segment: List[str] = [],
        sentence_start: bool = True,
        in_range: bool = False,
        range_start: bool = False,
        is_empty: Optional[bool] = None,
    ) -> None:
        object.__setattr__(self, "segment_ref", segment_ref)
        object.__setattr__(self, "segment", segment)
        object.__setattr__(self, "sentence_start", sentence_start)
        object.__setattr__(self, "is_in_range", in_range)
        object.__setattr__(self, "is_range_start", range_start)
        object.__setattr__(self, "is_empty", len(segment) == 0 if is_empty is None else is_empty)

    def __repr__(self) -> str:
        if self.is_empty:
            segment = "<range>" if self.is_in_range else "EMPTY"
        elif len(self.segment) > 0:
            segment = " ".join(self.segment)
        else:
            segment = "NONEMPTY"
        return f"{self.segment_ref} - {segment}"

from dataclasses import dataclass
from typing import Tuple

from .context_verse import ContextVerse


@dataclass(frozen=True)
class PostEditSegment:
    source: str
    draft: str
    context_verses: Tuple[ContextVerse, ...] = ()

    def __post_init__(self) -> None:
        offsets = [verse.offset_from_segment for verse in self.context_verses]
        if 0 in offsets:
            raise ValueError("A context verse cannot have offset 0; that offset is the segment itself.")
        if len(offsets) != len(set(offsets)):
            raise ValueError("Context verses must have distinct offsets.")

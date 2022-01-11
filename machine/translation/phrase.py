from dataclasses import dataclass

from ..annotations.range import Range


@dataclass(frozen=True)
class Phrase:
    source_segment_range: Range[int]
    target_segment_cut: int
    confidence: float

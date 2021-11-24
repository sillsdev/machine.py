from dataclasses import dataclass
from typing import Collection, Optional, Sequence

from .aligned_word_pair import AlignedWordPair


@dataclass(eq=False, frozen=True)
class ParallelTextSegment:
    text_id: str
    segment_ref: object
    source_segment: Sequence[str]
    target_segment: Sequence[str]
    aligned_word_pairs: Optional[Collection[AlignedWordPair]]
    is_source_sentence_start: bool
    is_source_in_range: bool
    is_source_range_start: bool
    is_target_sentence_start: bool
    is_target_in_range: bool
    is_target_range_start: bool
    is_empty: bool

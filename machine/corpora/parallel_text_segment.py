from dataclasses import dataclass
from typing import Any, Collection, Optional, Sequence

from .aligned_word_pair import AlignedWordPair


@dataclass(eq=False, frozen=True)
class ParallelTextSegment:
    text_id: str
    source_segment_ref: Optional[Any]
    target_segment_ref: Optional[Any]
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

    def __post_init__(self) -> None:
        if self.source_segment_ref is None and self.target_segment_ref is None:
            raise ValueError("Either the source or target segment ref must be set.")

    @property
    def segment_ref(self) -> Any:
        return self.target_segment_ref if self.source_segment_ref is None else self.source_segment_ref

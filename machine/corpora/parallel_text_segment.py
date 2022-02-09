from dataclasses import dataclass
from typing import Any, Collection, Optional, Sequence

from .aligned_word_pair import AlignedWordPair


@dataclass(eq=False, frozen=True)
class ParallelTextSegment:
    text_id: str
    source_segment_refs: Sequence[Any]
    target_segment_refs: Sequence[Any]
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
        if len(self.source_segment_refs) == 0 and len(self.target_segment_refs) == 0:
            raise ValueError("Either a source or target segment ref must be set.")

    @property
    def segment_ref(self) -> Any:
        return self.target_segment_refs[0] if len(self.source_segment_refs) == 0 else self.source_segment_refs[0]

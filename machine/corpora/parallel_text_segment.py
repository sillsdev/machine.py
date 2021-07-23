from dataclasses import dataclass
from typing import Collection, Optional, Sequence, cast

from .aligned_word_pair import AlignedWordPair
from .text_segment import TextSegment


@dataclass(eq=False, frozen=True)
class ParallelTextSegment:
    @classmethod
    def create(
        cls,
        text_id: str,
        source_segment: Optional[TextSegment],
        target_segment: Optional[TextSegment],
        aligned_word_pairs: Optional[Collection[AlignedWordPair]] = None,
    ) -> "ParallelTextSegment":
        return ParallelTextSegment(
            text_id,
            cast(TextSegment, target_segment).segment_ref if source_segment is None else source_segment.segment_ref,
            [] if source_segment is None else source_segment.segment,
            [] if target_segment is None else target_segment.segment,
            aligned_word_pairs,
            source_segment is not None and source_segment.is_in_range,
            source_segment is not None and source_segment.is_range_start,
            target_segment is not None and target_segment.is_in_range,
            target_segment is not None and target_segment.is_range_start,
            source_segment is None or source_segment.is_empty or target_segment is None or target_segment.is_empty,
        )

    @classmethod
    def create_range(
        cls, text_id: str, seg_ref: object, source_segment: Sequence[str], target_segment: Sequence[str], is_empty: bool
    ) -> "ParallelTextSegment":
        return ParallelTextSegment(
            text_id, seg_ref, source_segment, target_segment, None, False, False, False, False, is_empty
        )

    text_id: str
    segment_ref: object
    source_segment: Sequence[str]
    target_segment: Sequence[str]
    aligned_word_pairs: Optional[Collection[AlignedWordPair]]
    is_source_in_range: bool
    is_source_range_start: bool
    is_target_in_range: bool
    is_target_range_start: bool
    is_empty: bool

from dataclasses import dataclass, field
from typing import Collection, Iterable, List, Optional

from .aligned_word_pair import AlignedWordPair
from .parallel_text_segment import ParallelTextSegment
from .text import Text
from .text_alignment import TextAlignment
from .text_alignment_collection import TextAlignmentCollection
from .text_segment import TextSegment


def _check_same_ref_segments(same_ref_segments: List[TextSegment], other_segment: TextSegment) -> bool:
    if len(same_ref_segments) > 0 and same_ref_segments[0].segment_ref != other_segment.segment_ref:
        same_ref_segments.clear()

    return len(same_ref_segments) > 0


@dataclass
class RangeInfo:
    text: "ParallelText"
    segment_ref: Optional[object] = field(default=None, init=False)
    source_segment: List[str] = field(default_factory=list, init=False)
    target_segment: List[str] = field(default_factory=list, init=False)
    is_source_empty: bool = field(default=True, init=False)
    is_target_empty: bool = field(default=True, init=False)

    @property
    def is_in_range(self) -> bool:
        return self.segment_ref is not None

    def create_text_segment(self) -> ParallelTextSegment:
        seg = ParallelTextSegment.create_range(
            self.text.id,
            self.segment_ref,
            self.source_segment.copy(),
            self.target_segment.copy(),
            self.is_source_empty or self.is_target_empty,
        )
        self.segment_ref = None
        self.source_segment.clear()
        self.target_segment.clear()
        return seg


class ParallelText:
    def __init__(
        self, source_text: Text, target_text: Text, text_alignment_collection: Optional[TextAlignmentCollection] = None
    ) -> None:
        self._source_text = source_text
        self._target_text = target_text
        self._text_alignment_collection = text_alignment_collection

    @property
    def id(self) -> str:
        return self._source_text.id

    @property
    def sort_key(self) -> str:
        return self._source_text.sort_key

    @property
    def source_text(self) -> Text:
        return self._source_text

    @property
    def target_text(self) -> Text:
        return self._target_text

    @property
    def text_alignment_collection(self) -> Optional[TextAlignmentCollection]:
        return self._text_alignment_collection

    @property
    def segments(self) -> Iterable[ParallelTextSegment]:
        return self.get_segments()

    def get_segments(
        self, all_source_segments: bool = False, all_target_segments: bool = False, include_text: bool = True
    ) -> Iterable[ParallelTextSegment]:
        alignments = [] if self._text_alignment_collection is None else self._text_alignment_collection.alignments

        iterator1 = iter(self._source_text.get_segments(include_text))
        iterator2 = iter(self._target_text.get_segments(include_text))
        iterator3 = iter(alignments)

        range_info = RangeInfo(self)
        source_same_ref_segments: List[TextSegment] = []
        target_same_ref_segments: List[TextSegment] = []

        current1 = next(iterator1, None)
        current2 = next(iterator2, None)
        current3: Optional[TextAlignment] = None
        while current1 is not None and current2 is not None:
            if current1.segment_ref < current2.segment_ref:
                for seg in self._create_source_text_segments(
                    range_info, current1, target_same_ref_segments, all_source_segments
                ):
                    yield seg

                source_same_ref_segments.append(current1)
                current1 = next(iterator1, None)
            elif current1.segment_ref > current2.segment_ref:
                for seg in self._create_target_text_segments(
                    range_info, current2, source_same_ref_segments, all_target_segments
                ):
                    yield seg
                target_same_ref_segments.append(current2)
                current2 = next(iterator2, None)
            else:
                less_than = True
                while less_than:
                    current3 = next(iterator3, None)
                    if current3 is not None:
                        less_than = current1.segment_ref < current3.segment_ref
                    else:
                        less_than = False

                if (not all_target_segments and current1.is_in_range) or (
                    not all_source_segments and current2.is_in_range
                ):
                    if range_info.is_in_range and (
                        (current1.is_in_range and not current2.is_in_range and len(current1.segment) > 0)
                        or (not current1.is_in_range and current2.is_in_range and len(current2.segment) > 0)
                        or (
                            current1.is_in_range
                            and current2.is_in_range
                            and len(current1.segment) > 0
                            and len(current2.segment) > 0
                        )
                    ):
                        yield range_info.create_text_segment()

                    if not range_info.is_in_range:
                        range_info.segment_ref = current1.segment_ref
                    range_info.source_segment.extend(current1.segment)
                    range_info.target_segment.extend(current2.segment)
                    if range_info.is_source_empty:
                        range_info.is_source_empty = current1.is_empty
                    if range_info.is_target_empty:
                        range_info.is_target_empty = current2.is_empty
                else:
                    if _check_same_ref_segments(source_same_ref_segments, current2):
                        for prev_source_segment in source_same_ref_segments:
                            for seg in self._create_text_segments(range_info, prev_source_segment, current2):
                                yield seg

                    if _check_same_ref_segments(target_same_ref_segments, current1):
                        for prev_target_segment in target_same_ref_segments:
                            for seg in self._create_text_segments(range_info, current1, prev_target_segment):
                                yield seg

                    for seg in self._create_text_segments(
                        range_info,
                        current1,
                        current2,
                        current3.aligned_word_pairs
                        if current3 is not None and current1.segment_ref == current3.segment_ref
                        else None,
                    ):
                        yield seg

                source_same_ref_segments.append(current1)
                current1 = next(iterator1, None)

                target_same_ref_segments.append(current2)
                current2 = next(iterator2, None)

        while current1 is not None:
            for seg in self._create_source_text_segments(
                range_info, current1, target_same_ref_segments, all_source_segments
            ):
                yield seg
            current1 = next(iterator1, None)

        while current2 is not None:
            for seg in self._create_target_text_segments(
                range_info, current2, source_same_ref_segments, all_target_segments
            ):
                yield seg
            current2 = next(iterator2, None)

        if range_info.is_in_range:
            yield range_info.create_text_segment()

    def get_count(
        self, all_source_segments: bool = False, all_target_segments: bool = False, nonempty_only: bool = False
    ) -> int:
        return sum(
            1
            for s in self.get_segments(all_source_segments, all_target_segments, include_text=False)
            if not nonempty_only or not s.is_empty
        )

    def _create_text_segments(
        self,
        range_info: "RangeInfo",
        src_seg: Optional[TextSegment],
        trg_seg: Optional[TextSegment],
        aligned_word_pairs: Optional[Collection[AlignedWordPair]] = None,
    ) -> Iterable[ParallelTextSegment]:
        if range_info.is_in_range:
            yield range_info.create_text_segment()
        yield ParallelTextSegment.create(self.id, src_seg, trg_seg, aligned_word_pairs)

    def _create_source_text_segments(
        self,
        range_info: "RangeInfo",
        source_segment: TextSegment,
        target_same_ref_segments: List[TextSegment],
        all_source_segments: bool,
    ) -> Iterable[ParallelTextSegment]:
        if _check_same_ref_segments(target_same_ref_segments, source_segment):
            for target_same_ref_segment in target_same_ref_segments:
                for seg in self._create_text_segments(range_info, source_segment, target_same_ref_segment):
                    yield seg
        elif all_source_segments:
            for seg in self._create_text_segments(range_info, source_segment, None):
                yield seg

    def _create_target_text_segments(
        self,
        range_info: "RangeInfo",
        target_segment: TextSegment,
        source_same_ref_segments: List[TextSegment],
        all_target_segments: bool,
    ) -> Iterable[ParallelTextSegment]:
        if _check_same_ref_segments(source_same_ref_segments, target_segment):
            for source_same_ref_segment in source_same_ref_segments:
                for seg in self._create_text_segments(range_info, source_same_ref_segment, target_segment):
                    yield seg
        elif all_target_segments:
            for seg in self._create_text_segments(range_info, None, target_segment):
                yield seg

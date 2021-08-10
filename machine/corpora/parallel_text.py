from dataclasses import dataclass, field
from typing import Collection, Generator, Iterable, List, Optional, cast

from ..utils.comparable import compare
from ..utils.context_managed_generator import ContextManagedGenerator
from .aligned_word_pair import AlignedWordPair
from .null_text_alignment_collection import NullTextAlignmentCollection
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
        seg = ParallelTextSegment(
            self.text.id,
            self.segment_ref,
            self.source_segment.copy(),
            self.target_segment.copy(),
            aligned_word_pairs=None,
            is_source_in_range=False,
            is_source_range_start=False,
            is_target_in_range=False,
            is_target_range_start=False,
            is_empty=self.is_source_empty or self.is_target_empty,
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
        self._text_alignment_collection = (
            NullTextAlignmentCollection(source_text.id, source_text.sort_key)
            if text_alignment_collection is None
            else text_alignment_collection
        )

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
    def text_alignment_collection(self) -> TextAlignmentCollection:
        return self._text_alignment_collection

    @property
    def segments(self) -> ContextManagedGenerator[ParallelTextSegment, None, None]:
        return self.get_segments()

    def get_segments(
        self, all_source_segments: bool = False, all_target_segments: bool = False, include_text: bool = True
    ) -> ContextManagedGenerator[ParallelTextSegment, None, None]:
        return ContextManagedGenerator(self._get_segments(all_source_segments, all_target_segments, include_text))

    def _get_segments(
        self, all_source_segments: bool, all_target_segments: bool, include_text: bool
    ) -> Generator[ParallelTextSegment, None, None]:
        with self._source_text.get_segments(include_text) as src_iterator, self._target_text.get_segments(
            include_text
        ) as trg_iterator, self._text_alignment_collection.alignments as alignment_iterator:
            range_info = RangeInfo(self)
            source_same_ref_segments: List[TextSegment] = []
            target_same_ref_segments: List[TextSegment] = []

            src_segment = next(src_iterator, None)
            trg_segment = next(trg_iterator, None)
            alignment: Optional[TextAlignment] = None
            while src_segment is not None and trg_segment is not None:
                compare1 = compare(src_segment.segment_ref, trg_segment.segment_ref)
                if compare1 < 0:
                    for seg in self._create_source_text_segments(
                        range_info, src_segment, target_same_ref_segments, all_source_segments
                    ):
                        yield seg

                    source_same_ref_segments.append(src_segment)
                    src_segment = next(src_iterator, None)
                elif compare1 > 0:
                    for seg in self._create_target_text_segments(
                        range_info, trg_segment, source_same_ref_segments, all_target_segments
                    ):
                        yield seg
                    target_same_ref_segments.append(trg_segment)
                    trg_segment = next(trg_iterator, None)
                else:
                    compare2 = -1
                    while compare2 < 0:
                        alignment = next(alignment_iterator, None)
                        compare2 = 1 if alignment is None else compare(src_segment.segment_ref, alignment.segment_ref)

                    if (not all_target_segments and src_segment.is_in_range) or (
                        not all_source_segments and trg_segment.is_in_range
                    ):
                        if range_info.is_in_range and (
                            (src_segment.is_in_range and not trg_segment.is_in_range and len(src_segment.segment) > 0)
                            or (
                                not src_segment.is_in_range and trg_segment.is_in_range and len(trg_segment.segment) > 0
                            )
                            or (
                                src_segment.is_in_range
                                and trg_segment.is_in_range
                                and len(src_segment.segment) > 0
                                and len(trg_segment.segment) > 0
                            )
                        ):
                            yield range_info.create_text_segment()

                        if not range_info.is_in_range:
                            range_info.segment_ref = src_segment.segment_ref
                        range_info.source_segment.extend(src_segment.segment)
                        range_info.target_segment.extend(trg_segment.segment)
                        if range_info.is_source_empty:
                            range_info.is_source_empty = src_segment.is_empty
                        if range_info.is_target_empty:
                            range_info.is_target_empty = trg_segment.is_empty
                    else:
                        if _check_same_ref_segments(source_same_ref_segments, trg_segment):
                            for prev_source_segment in source_same_ref_segments:
                                for seg in self._create_text_segments(range_info, prev_source_segment, trg_segment):
                                    yield seg

                        if _check_same_ref_segments(target_same_ref_segments, src_segment):
                            for prev_target_segment in target_same_ref_segments:
                                for seg in self._create_text_segments(range_info, src_segment, prev_target_segment):
                                    yield seg

                        for seg in self._create_text_segments(
                            range_info,
                            src_segment,
                            trg_segment,
                            alignment.aligned_word_pairs
                            if alignment is not None and src_segment.segment_ref == alignment.segment_ref
                            else None,
                        ):
                            yield seg

                    source_same_ref_segments.append(src_segment)
                    src_segment = next(src_iterator, None)

                    target_same_ref_segments.append(trg_segment)
                    trg_segment = next(trg_iterator, None)

            while src_segment is not None:
                for seg in self._create_source_text_segments(
                    range_info, src_segment, target_same_ref_segments, all_source_segments
                ):
                    yield seg
                src_segment = next(src_iterator, None)

            while trg_segment is not None:
                for seg in self._create_target_text_segments(
                    range_info, trg_segment, source_same_ref_segments, all_target_segments
                ):
                    yield seg
                trg_segment = next(trg_iterator, None)

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
        yield ParallelTextSegment(
            self.id,
            cast(TextSegment, trg_seg).segment_ref if src_seg is None else src_seg.segment_ref,
            [] if src_seg is None else src_seg.segment,
            [] if trg_seg is None else trg_seg.segment,
            aligned_word_pairs,
            src_seg is not None and src_seg.is_in_range,
            src_seg is not None and src_seg.is_range_start,
            trg_seg is not None and trg_seg.is_in_range,
            trg_seg is not None and trg_seg.is_range_start,
            src_seg is None or src_seg.is_empty or trg_seg is None or trg_seg.is_empty,
        )

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

from dataclasses import dataclass, field
from typing import Any, Collection, Generator, Iterable, List, Optional

from ..scripture.verse_ref import VerseRef
from ..utils.comparable import compare
from .aligned_word_pair import AlignedWordPair
from .dictionary_text_alignment_corpus import DictionaryTextAlignmentCorpus
from .parallel_text_corpus_row import ParallelTextCorpusRow
from .parallel_text_corpus_view import ParallelTextCorpusView
from .text_alignment_corpus_row import TextAlignmentCorpusRow
from .text_alignment_corpus_view import TextAlignmentCorpusView
from .text_corpus_row import TextCorpusRow
from .text_corpus_view import TextCorpusView


class ParallelTextCorpus(ParallelTextCorpusView):
    def __init__(
        self,
        source_corpus: TextCorpusView,
        target_corpus: TextCorpusView,
        text_alignment_corpus: Optional[TextAlignmentCorpusView] = None,
    ) -> None:
        self._source_corpus = source_corpus
        self._target_corpus = target_corpus
        self._text_alignment_corpus = (
            DictionaryTextAlignmentCorpus() if text_alignment_corpus is None else text_alignment_corpus
        )

    @property
    def source(self) -> ParallelTextCorpusView:
        return self

    def _get_rows(self, all_source_rows: bool, all_target_rows: bool) -> Generator[ParallelTextCorpusRow, None, None]:
        with self._source_corpus.get_rows() as src_iterator, self._target_corpus.get_rows(
            self._source_corpus
        ) as trg_iterator, self._text_alignment_corpus.get_rows() as alignment_iterator:
            range_info = RangeInfo()
            source_same_ref_rows: List[TextCorpusRow] = []
            target_same_ref_rows: List[TextCorpusRow] = []

            src_row = next(src_iterator, None)
            trg_row = next(trg_iterator, None)
            alignment: Optional[TextAlignmentCorpusRow] = None
            while src_row is not None and trg_row is not None:
                compare1 = _compare_refs(src_row.ref, trg_row.ref)
                if compare1 < 0:
                    if not all_target_rows and src_row.is_in_range:
                        if range_info.is_in_range and trg_row.is_in_range and len(trg_row.segment) > 0:
                            yield range_info.create_row()
                        range_info.source_refs.append(src_row.ref)
                        target_same_ref_rows.clear()
                        range_info.source_segment.extend(src_row.segment)
                        if range_info.is_source_empty:
                            range_info.is_source_empty = src_row.is_empty
                            range_info.is_source_sentence_start = src_row.is_sentence_start
                    else:
                        yield from self._create_source_rows(range_info, src_row, target_same_ref_rows, all_source_rows)

                    source_same_ref_rows.append(src_row)
                    src_row = next(src_iterator, None)
                elif compare1 > 0:
                    if not all_source_rows and trg_row.is_in_range:
                        if range_info.is_in_range and src_row.is_in_range and len(src_row.segment) > 0:
                            yield range_info.create_row()
                        range_info.target_refs.append(trg_row.ref)
                        source_same_ref_rows.clear()
                        range_info.target_segment.extend(trg_row.segment)
                        if range_info.is_target_empty:
                            range_info.is_target_empty = trg_row.is_empty
                            range_info.is_target_sentence_start = trg_row.is_sentence_start
                    else:
                        yield from self._create_target_rows(range_info, trg_row, source_same_ref_rows, all_target_rows)
                    target_same_ref_rows.append(trg_row)
                    trg_row = next(trg_iterator, None)
                else:
                    compare2 = -1
                    while compare2 < 0:
                        alignment = next(alignment_iterator, None)
                        compare2 = 1 if alignment is None else _compare_refs(src_row.ref, alignment.ref)

                    if (not all_target_rows and src_row.is_in_range) or (not all_source_rows and trg_row.is_in_range):
                        if range_info.is_in_range and (
                            (src_row.is_in_range and not trg_row.is_in_range and len(src_row.segment) > 0)
                            or (not src_row.is_in_range and trg_row.is_in_range and len(trg_row.segment) > 0)
                            or (
                                src_row.is_in_range
                                and trg_row.is_in_range
                                and len(src_row.segment) > 0
                                and len(trg_row.segment) > 0
                            )
                        ):
                            yield range_info.create_row()

                        range_info.source_refs.append(src_row.ref)
                        range_info.target_refs.append(trg_row.ref)
                        source_same_ref_rows.clear()
                        target_same_ref_rows.clear()
                        range_info.source_segment.extend(src_row.segment)
                        range_info.target_segment.extend(trg_row.segment)
                        if range_info.is_source_empty:
                            range_info.is_source_empty = src_row.is_empty
                            range_info.is_source_sentence_start = src_row.is_sentence_start
                        if range_info.is_target_empty:
                            range_info.is_target_empty = trg_row.is_empty
                            range_info.is_target_sentence_start = trg_row.is_sentence_start
                    else:
                        if _check_same_ref_rows(source_same_ref_rows, trg_row):
                            for prev_source_segment in source_same_ref_rows:
                                yield from self._create_rows(range_info, prev_source_segment, trg_row)

                        if _check_same_ref_rows(target_same_ref_rows, src_row):
                            for prev_target_segment in target_same_ref_rows:
                                yield from self._create_rows(range_info, src_row, prev_target_segment)

                        yield from self._create_rows(
                            range_info,
                            src_row,
                            trg_row,
                            alignment.aligned_word_pairs
                            if alignment is not None and src_row.ref == alignment.ref
                            else None,
                        )

                    source_same_ref_rows.append(src_row)
                    src_row = next(src_iterator, None)

                    target_same_ref_rows.append(trg_row)
                    trg_row = next(trg_iterator, None)

            while src_row is not None:
                yield from self._create_source_rows(range_info, src_row, target_same_ref_rows, all_source_rows)
                src_row = next(src_iterator, None)

            while trg_row is not None:
                yield from self._create_target_rows(range_info, trg_row, source_same_ref_rows, all_target_rows)
                trg_row = next(trg_iterator, None)

            if range_info.is_in_range:
                yield range_info.create_row()

    def _create_rows(
        self,
        range_info: "RangeInfo",
        src_row: Optional[TextCorpusRow],
        trg_row: Optional[TextCorpusRow],
        aligned_word_pairs: Optional[Collection[AlignedWordPair]] = None,
    ) -> Iterable[ParallelTextCorpusRow]:
        if range_info.is_in_range:
            yield range_info.create_row()
        yield ParallelTextCorpusRow(
            [] if src_row is None else [src_row.ref],
            [] if trg_row is None else [trg_row.ref],
            [] if src_row is None else src_row.segment,
            [] if trg_row is None else trg_row.segment,
            aligned_word_pairs,
            src_row is not None and src_row.is_sentence_start,
            src_row is not None and src_row.is_in_range,
            src_row is not None and src_row.is_range_start,
            trg_row is not None and trg_row.is_sentence_start,
            trg_row is not None and trg_row.is_in_range,
            trg_row is not None and trg_row.is_range_start,
            src_row is None or src_row.is_empty or trg_row is None or trg_row.is_empty,
        )

    def _create_source_rows(
        self,
        range_info: "RangeInfo",
        source_row: TextCorpusRow,
        target_same_ref_rows: List[TextCorpusRow],
        all_source_rows: bool,
    ) -> Iterable[ParallelTextCorpusRow]:
        if _check_same_ref_rows(target_same_ref_rows, source_row):
            for target_same_ref_segment in target_same_ref_rows:
                for seg in self._create_rows(range_info, source_row, target_same_ref_segment):
                    yield seg
        elif all_source_rows:
            for seg in self._create_rows(range_info, source_row, None):
                yield seg

    def _create_target_rows(
        self,
        range_info: "RangeInfo",
        target_row: TextCorpusRow,
        source_same_ref_rows: List[TextCorpusRow],
        all_target_rows: bool,
    ) -> Iterable[ParallelTextCorpusRow]:
        if _check_same_ref_rows(source_same_ref_rows, target_row):
            for source_same_ref_segment in source_same_ref_rows:
                for seg in self._create_rows(range_info, source_same_ref_segment, target_row):
                    yield seg
        elif all_target_rows:
            for seg in self._create_rows(range_info, None, target_row):
                yield seg


@dataclass
class RangeInfo:
    source_refs: List[Any] = field(default_factory=list, init=False)
    target_refs: List[Any] = field(default_factory=list, init=False)
    source_segment: List[str] = field(default_factory=list, init=False)
    target_segment: List[str] = field(default_factory=list, init=False)
    is_source_sentence_start: bool = field(default=False, init=False)
    is_target_sentence_start: bool = field(default=False, init=False)
    is_source_empty: bool = field(default=True, init=False)
    is_target_empty: bool = field(default=True, init=False)

    @property
    def is_in_range(self) -> bool:
        return len(self.source_refs) > 0 and len(self.target_refs) > 0

    def create_row(self) -> ParallelTextCorpusRow:
        seg = ParallelTextCorpusRow(
            self.source_refs.copy(),
            self.target_refs.copy(),
            self.source_segment.copy(),
            self.target_segment.copy(),
            aligned_word_pairs=None,
            is_source_sentence_start=self.is_source_sentence_start,
            is_source_in_range=False,
            is_source_range_start=False,
            is_target_sentence_start=self.is_target_sentence_start,
            is_target_in_range=False,
            is_target_range_start=False,
            is_empty=self.is_source_empty or self.is_target_empty,
        )
        self.source_refs.clear()
        self.target_refs.clear()
        self.source_segment.clear()
        self.target_segment.clear()
        self.is_source_sentence_start = False
        self.is_target_sentence_start = False
        self.is_source_empty = True
        self.is_target_empty = True
        return seg


def _check_same_ref_rows(same_ref_rows: List[TextCorpusRow], other_row: TextCorpusRow) -> bool:
    if len(same_ref_rows) > 0 and _compare_refs(same_ref_rows[0].ref, other_row.ref) != 0:
        same_ref_rows.clear()

    return len(same_ref_rows) > 0


def _compare_refs(source_ref: Any, target_ref: Any) -> int:
    if isinstance(source_ref, VerseRef) and isinstance(target_ref, VerseRef):
        return source_ref.compare_to(target_ref, compare_segments=False)
    return compare(source_ref, target_ref)

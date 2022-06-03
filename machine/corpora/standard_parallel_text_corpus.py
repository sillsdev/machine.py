from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Collection, ContextManager, Generator, Iterable, List, Optional, Set, Tuple

from ..scripture.verse_ref import VerseRef, Versification
from ..utils.comparable import compare
from ..utils.context_managed_generator import ContextManagedGenerator
from .aligned_word_pair import AlignedWordPair
from .alignment_corpus import AlignmentCorpus
from .alignment_row import AlignmentRow
from .dictionary_alignment_corpus import DictionaryAlignmentCorpus
from .parallel_text_corpus import ParallelTextCorpus
from .parallel_text_row import ParallelTextRow
from .text_corpus import TextCorpus
from .text_row import TextRow


class StandardParallelTextCorpus(ParallelTextCorpus):
    def __init__(
        self,
        source_corpus: TextCorpus,
        target_corpus: TextCorpus,
        alignment_corpus: Optional[AlignmentCorpus] = None,
        all_source_rows: bool = False,
        all_target_rows: bool = False,
    ) -> None:
        self._source_corpus = source_corpus
        self._target_corpus = target_corpus
        self._alignment_corpus = DictionaryAlignmentCorpus() if alignment_corpus is None else alignment_corpus
        self._all_source_rows = all_source_rows
        self._all_target_rows = all_target_rows

    @property
    def source_corpus(self) -> TextCorpus:
        return self._source_corpus

    @property
    def target_corpus(self) -> TextCorpus:
        return self._target_corpus

    @property
    def alignment_corpus(self) -> AlignmentCorpus:
        return self._alignment_corpus

    @property
    def all_source_rows(self) -> bool:
        return self._all_source_rows

    @property
    def all_target_rows(self) -> bool:
        return self._all_target_rows

    @property
    def missing_rows_allowed(self) -> bool:
        if self._source_corpus.missing_rows_allowed or self._target_corpus.missing_rows_allowed:
            return True
        source_text_ids = {t.id for t in self._source_corpus.texts}
        target_text_ids = {t.id for t in self._target_corpus.texts}
        return source_text_ids != target_text_ids

    def count(self, include_empty: bool = True) -> int:
        if self.missing_rows_allowed:
            return super().count(include_empty)
        if include_empty:
            return self._source_corpus.count(include_empty)
        return min(self._source_corpus.count(include_empty), self._target_corpus.count(include_empty))

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        source_text_ids = {t.id for t in self._source_corpus.texts}
        target_text_ids = {t.id for t in self._target_corpus.texts}

        text_ids: Set[str]
        if self._all_source_rows and self._all_target_rows:
            text_ids = source_text_ids | target_text_ids
        elif not self._all_source_rows and not self._all_target_rows:
            text_ids = source_text_ids & target_text_ids
        elif self._all_source_rows:
            text_ids = source_text_ids
        else:
            text_ids = target_text_ids

        with ExitStack() as stack:
            src_iterator = stack.enter_context(self._source_corpus.get_rows(text_ids))
            trg_iterator = stack.enter_context(_TargetCorpusGenerator(self._target_corpus.get_rows(text_ids)))
            alignment_iterator = stack.enter_context(self._alignment_corpus.get_rows(text_ids))

            range_info = _RangeInfo()
            source_same_ref_rows: List[TextRow] = []
            target_same_ref_rows: List[TextRow] = []

            src_row = next(src_iterator, None)
            if src_row is not None and isinstance(src_row.ref, VerseRef):
                trg_iterator.source_versification = src_row.ref.versification
            trg_row = next(trg_iterator, None)
            alignment: Optional[AlignmentRow] = None
            while src_row is not None and trg_row is not None:
                compare1 = _compare_refs(src_row.ref, trg_row.ref)
                if compare1 < 0:
                    if not self._all_target_rows and src_row.is_in_range:
                        if range_info.is_in_range and trg_row.is_in_range and len(trg_row.segment) > 0:
                            yield range_info.create_row()
                        range_info.source_refs.append(src_row.ref)
                        target_same_ref_rows.clear()
                        range_info.source_segment.extend(src_row.segment)
                        if range_info.is_source_empty:
                            range_info.is_source_empty = src_row.is_empty
                            range_info.is_source_sentence_start = src_row.is_sentence_start
                    else:
                        yield from self._create_source_rows(range_info, src_row, target_same_ref_rows)

                    source_same_ref_rows.append(src_row)
                    src_row = next(src_iterator, None)
                elif compare1 > 0:
                    if not self._all_source_rows and trg_row.is_in_range:
                        if range_info.is_in_range and src_row.is_in_range and len(src_row.segment) > 0:
                            yield range_info.create_row()
                        range_info.target_refs.append(trg_row.ref)
                        source_same_ref_rows.clear()
                        range_info.target_segment.extend(trg_row.segment)
                        if range_info.is_target_empty:
                            range_info.is_target_empty = trg_row.is_empty
                            range_info.is_target_sentence_start = trg_row.is_sentence_start
                    else:
                        yield from self._create_target_rows(range_info, trg_row, source_same_ref_rows)
                    target_same_ref_rows.append(trg_row)
                    trg_row = next(trg_iterator, None)
                else:
                    compare2 = -1
                    while compare2 < 0:
                        alignment = next(alignment_iterator, None)
                        compare2 = 1 if alignment is None else _compare_refs(src_row.ref, alignment.ref)

                    if (not self._all_target_rows and src_row.is_in_range) or (
                        not self._all_source_rows and trg_row.is_in_range
                    ):
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

                        range_info.text_id = src_row.text_id
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
                if not self._all_target_rows and src_row.is_in_range:
                    range_info.text_id = src_row.text_id
                    range_info.source_refs.append(src_row.ref)
                    target_same_ref_rows.clear()
                    range_info.source_segment.extend(src_row.segment)
                    if range_info.is_source_empty:
                        range_info.is_source_empty = src_row.is_empty
                        range_info.is_source_sentence_start = src_row.is_sentence_start
                else:
                    yield from self._create_source_rows(range_info, src_row, target_same_ref_rows)
                src_row = next(src_iterator, None)

            while trg_row is not None:
                if not self._all_source_rows and trg_row.is_in_range:
                    range_info.text_id = trg_row.text_id
                    range_info.target_refs.append(trg_row.ref)
                    source_same_ref_rows.clear()
                    range_info.target_segment.extend(trg_row.segment)
                    if range_info.is_target_empty:
                        range_info.is_target_empty = trg_row.is_empty
                        range_info.is_target_sentence_start = trg_row.is_sentence_start
                else:
                    yield from self._create_target_rows(range_info, trg_row, source_same_ref_rows)
                trg_row = next(trg_iterator, None)

            if range_info.is_in_range:
                yield range_info.create_row()

    def _create_rows(
        self,
        range_info: _RangeInfo,
        src_row: Optional[TextRow],
        trg_row: Optional[TextRow],
        aligned_word_pairs: Optional[Collection[AlignedWordPair]] = None,
    ) -> Iterable[ParallelTextRow]:
        if range_info.is_in_range:
            yield range_info.create_row()
        if src_row is not None:
            text_id = src_row.text_id
        elif trg_row is not None:
            text_id = trg_row.text_id
        else:
            raise ValueError("Either a source or target must be specified.")
        yield ParallelTextRow(
            text_id,
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
        range_info: _RangeInfo,
        source_row: TextRow,
        target_same_ref_rows: List[TextRow],
    ) -> Iterable[ParallelTextRow]:
        if _check_same_ref_rows(target_same_ref_rows, source_row):
            for target_same_ref_segment in target_same_ref_rows:
                for seg in self._create_rows(range_info, source_row, target_same_ref_segment):
                    yield seg
        elif self._all_source_rows:
            for seg in self._create_rows(range_info, source_row, None):
                yield seg

    def _create_target_rows(
        self,
        range_info: _RangeInfo,
        target_row: TextRow,
        source_same_ref_rows: List[TextRow],
    ) -> Iterable[ParallelTextRow]:
        if _check_same_ref_rows(source_same_ref_rows, target_row):
            for source_same_ref_segment in source_same_ref_rows:
                for seg in self._create_rows(range_info, source_same_ref_segment, target_row):
                    yield seg
        elif self._all_target_rows:
            for seg in self._create_rows(range_info, None, target_row):
                yield seg


@dataclass
class _RangeInfo:
    text_id: str = field(default="", init=False)
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

    def create_row(self) -> ParallelTextRow:
        row = ParallelTextRow(
            self.text_id,
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
        self.text_id = ""
        self.source_refs.clear()
        self.target_refs.clear()
        self.source_segment.clear()
        self.target_segment.clear()
        self.is_source_sentence_start = False
        self.is_target_sentence_start = False
        self.is_source_empty = True
        self.is_target_empty = True
        return row


class _TargetCorpusGenerator(ContextManager["_TargetCorpusGenerator"], Generator[TextRow, None, None]):
    def __init__(self, generator: ContextManagedGenerator[TextRow, None, None]) -> None:
        self._generator = generator
        self._is_scripture = False
        self._is_enumerating = False
        self._verse_rows: SimpleQueue[TextRow] = SimpleQueue()
        self.source_versification: Optional[Versification] = None
        self._row: Optional[TextRow] = None

    def send(self, value: None) -> TextRow:
        if not self._is_enumerating:
            self._is_enumerating = True
            self._row = next(self._generator, None)
            if (
                self._row is not None
                and isinstance(self._row.ref, VerseRef)
                and self.source_versification != self._row.ref.versification
            ):
                self._is_scripture = True
            elif self._row is not None:
                return self._row
            else:
                raise StopIteration

        if self._is_scripture:
            if self._verse_rows.empty():
                self._collect_verses()
            if not self._verse_rows.empty():
                return self._verse_rows.get()
            raise StopIteration

        self._row = next(self._generator, None)
        if self._row is not None:
            return self._row
        raise StopIteration

    def throw(self, type: Any, value: Any = None, traceback: Any = None) -> TextRow:
        raise StopIteration

    def close(self) -> None:
        super().close()
        self._generator.close()

    def __enter__(self) -> _TargetCorpusGenerator:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.close()

    def _collect_verses(self) -> None:
        assert self.source_versification is not None
        seg_list: List[Tuple[VerseRef, TextRow]] = []
        out_of_order = False
        prev_verse_ref = VerseRef()
        range_start_offset = -1
        while self._row is not None:
            row = self._row
            verse_ref: VerseRef = row.ref
            verse_ref = verse_ref.copy()
            verse_ref.change_versification(self.source_versification)
            # convert one-to-many mapping to a verse range
            if verse_ref == prev_verse_ref:
                range_start_verse_ref, range_start_row = seg_list[range_start_offset]
                is_range_start = False
                if range_start_offset == -1:
                    is_range_start = range_start_row.is_range_start if range_start_row.is_in_range else True
                seg_list[range_start_offset] = (
                    range_start_verse_ref,
                    TextRow(
                        range_start_row.text_id,
                        range_start_row.ref,
                        list(range_start_row.segment) + list(row.segment),
                        range_start_row.is_sentence_start,
                        is_in_range=True,
                        is_range_start=is_range_start,
                        is_empty=range_start_row.is_empty and row.is_empty,
                    ),
                )
                row = TextRow(row.text_id, row.ref, is_in_range=True)
                range_start_offset -= 1
            else:
                range_start_offset = -1
            seg_list.append((verse_ref, row))
            if not out_of_order and verse_ref < prev_verse_ref:
                out_of_order = True
            prev_verse_ref = verse_ref
            self._row = next(self._generator, None)

        if out_of_order:
            seg_list.sort(key=lambda t: t[0])

        for _, row in seg_list:
            self._verse_rows.put(row)


def _check_same_ref_rows(same_ref_rows: List[TextRow], other_row: TextRow) -> bool:
    if len(same_ref_rows) > 0 and _compare_refs(same_ref_rows[0].ref, other_row.ref) != 0:
        same_ref_rows.clear()

    return len(same_ref_rows) > 0


def _compare_refs(source_ref: Any, target_ref: Any) -> int:
    if isinstance(source_ref, VerseRef) and isinstance(target_ref, VerseRef):
        return source_ref.compare_to(target_ref, compare_segments=False)
    return compare(source_ref, target_ref)

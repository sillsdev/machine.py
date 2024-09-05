from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Collection, ContextManager, Generator, Iterable, List, Optional, Set, Tuple

from ..scripture.verse_ref import Versification
from ..utils.comparable import compare
from ..utils.context_managed_generator import ContextManagedGenerator
from .aligned_word_pair import AlignedWordPair
from .alignment_corpus import AlignmentCorpus
from .alignment_row import AlignmentRow
from .dictionary_alignment_corpus import DictionaryAlignmentCorpus
from .parallel_text_corpus import ParallelTextCorpus
from .parallel_text_row import ParallelTextRow
from .scripture_ref import EMPTY_SCRIPTURE_REF, ScriptureRef
from .scripture_text_corpus import is_scripture
from .text_corpus import TextCorpus
from .text_row import TextRow, TextRowFlags


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
    def is_source_tokenized(self) -> bool:
        return self.source_corpus.is_tokenized

    @property
    def is_target_tokenized(self) -> bool:
        return self.target_corpus.is_tokenized

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

    def _get_rows(self, text_ids: Optional[Iterable[str]]) -> Generator[ParallelTextRow, None, None]:
        source_text_ids = {t.id for t in self._source_corpus.texts}
        target_text_ids = {t.id for t in self._target_corpus.texts}

        filter_text_ids: Set[str]
        if self._all_source_rows and self._all_target_rows:
            filter_text_ids = set(source_text_ids)
            filter_text_ids.update(target_text_ids)
        elif not self._all_source_rows and not self._all_target_rows:
            filter_text_ids = set(source_text_ids)
            filter_text_ids.intersection_update(target_text_ids)
        elif self._all_source_rows:
            filter_text_ids = set(source_text_ids)
        else:
            filter_text_ids = set(target_text_ids)
        if text_ids is not None:
            filter_text_ids.intersection_update(text_ids)

        with ExitStack() as stack:
            src_iterator = stack.enter_context(self._source_corpus.get_rows(filter_text_ids))
            trg_iterator = stack.enter_context(
                _TargetCorpusGenerator(
                    self._target_corpus.get_rows(filter_text_ids),
                    self._source_corpus.versification,
                    self._target_corpus.versification,
                )
            )
            alignment_iterator = stack.enter_context(self._alignment_corpus.get_rows(filter_text_ids))

            range_info = _RangeInfo(target_versification=self._target_corpus.versification)
            source_same_ref_rows: List[TextRow] = []
            target_same_ref_rows: List[TextRow] = []

            src_row = next(src_iterator, None)
            trg_row = next(trg_iterator, None)
            alignment: Optional[AlignmentRow] = None
            while src_row is not None and trg_row is not None:
                compare1 = _compare_refs(src_row.ref, trg_row.ref)
                if compare1 < 0:
                    if not self._all_target_rows and src_row.is_in_range:
                        if range_info.is_in_range and trg_row.is_in_range and len(trg_row.segment) > 0:
                            yield range_info.create_row()
                        range_info.text_id = src_row.text_id
                        range_info.source_refs.append(src_row.ref)
                        target_same_ref_rows.clear()
                        range_info.source_segment.extend(src_row.segment)
                        if range_info.is_source_empty:
                            range_info.is_source_empty = src_row.is_empty
                            range_info.is_source_sentence_start = src_row.is_sentence_start
                    else:
                        yield from self._create_source_rows(
                            range_info,
                            src_row,
                            target_same_ref_rows,
                            force_target_in_range=src_row.text_id == trg_row.text_id
                            and not trg_row.is_range_start
                            and trg_row.is_in_range,
                        )

                    source_same_ref_rows.append(src_row)
                    src_row = next(src_iterator, None)
                elif compare1 > 0:
                    if not self._all_source_rows and trg_row.is_in_range:
                        if range_info.is_in_range and src_row.is_in_range and len(src_row.segment) > 0:
                            yield range_info.create_row()
                        range_info.text_id = trg_row.text_id
                        range_info.target_refs.append(trg_row.ref)
                        source_same_ref_rows.clear()
                        range_info.target_segment.extend(trg_row.segment)
                        if range_info.is_target_empty:
                            range_info.is_target_empty = trg_row.is_empty
                            range_info.is_target_sentence_start = trg_row.is_sentence_start
                    else:
                        yield from self._create_target_rows(
                            range_info,
                            trg_row,
                            source_same_ref_rows,
                            force_source_in_range=trg_row.text_id == src_row.text_id
                            and not src_row.is_range_start
                            and src_row.is_in_range,
                        )
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
                            (
                                alignment.aligned_word_pairs
                                if alignment is not None and src_row.ref == alignment.ref
                                else None
                            ),
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
        force_source_in_range: bool = False,
        force_target_in_range: bool = False,
    ) -> Iterable[ParallelTextRow]:
        if range_info.is_in_range:
            yield range_info.create_row()
        if src_row is not None:
            text_id = src_row.text_id
        elif trg_row is not None:
            text_id = trg_row.text_id
        else:
            raise ValueError("Either a source or target must be specified.")

        src_refs = [] if src_row is None else [src_row.ref]
        trg_refs = [] if trg_row is None else [trg_row.ref]

        if len(trg_refs) == 0 and is_scripture(self._target_corpus):
            for r in src_refs:
                r: ScriptureRef
                trg_refs.append(r.change_versification(self._target_corpus.versification))

        if src_row is None:
            source_flags = TextRowFlags.IN_RANGE if force_source_in_range else TextRowFlags.NONE
        else:
            source_flags = src_row.flags

        if trg_row is None:
            target_flags = TextRowFlags.IN_RANGE if force_target_in_range else TextRowFlags.NONE
        else:
            target_flags = trg_row.flags

        yield ParallelTextRow(
            text_id,
            src_refs,
            trg_refs,
            [] if src_row is None else src_row.segment,
            [] if trg_row is None else trg_row.segment,
            aligned_word_pairs,
            source_flags,
            target_flags,
        )

    def _create_source_rows(
        self,
        range_info: _RangeInfo,
        source_row: TextRow,
        target_same_ref_rows: List[TextRow],
        force_target_in_range: bool = False,
    ) -> Iterable[ParallelTextRow]:
        if _check_same_ref_rows(target_same_ref_rows, source_row):
            for target_same_ref_segment in target_same_ref_rows:
                yield from self._create_rows(range_info, source_row, target_same_ref_segment)
        elif self._all_source_rows:
            yield from self._create_rows(range_info, source_row, None, force_target_in_range=force_target_in_range)

    def _create_target_rows(
        self,
        range_info: _RangeInfo,
        target_row: TextRow,
        source_same_ref_rows: List[TextRow],
        force_source_in_range: bool = False,
    ) -> Iterable[ParallelTextRow]:
        if _check_same_ref_rows(source_same_ref_rows, target_row):
            for source_same_ref_segment in source_same_ref_rows:
                yield from self._create_rows(range_info, source_same_ref_segment, target_row)
        elif self._all_target_rows:
            yield from self._create_rows(range_info, None, target_row, force_source_in_range=force_source_in_range)


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
    target_versification: Optional[Versification] = field(default=None)

    @property
    def is_in_range(self) -> bool:
        return len(self.source_refs) > 0 or len(self.target_refs) > 0

    def create_row(self) -> ParallelTextRow:
        if len(self.target_refs) == 0 and self.target_versification is not None:
            for r in self.source_refs:
                r: ScriptureRef
                self.target_refs.append(r.change_versification(self.target_versification))
        row = ParallelTextRow(
            self.text_id,
            self.source_refs.copy(),
            self.target_refs.copy(),
            self.source_segment.copy(),
            self.target_segment.copy(),
            aligned_word_pairs=None,
            source_flags=TextRowFlags.SENTENCE_START if self.is_source_sentence_start else TextRowFlags.NONE,
            target_flags=TextRowFlags.SENTENCE_START if self.is_target_sentence_start else TextRowFlags.NONE,
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
    def __init__(
        self,
        generator: ContextManagedGenerator[TextRow, None, None],
        source_versification: Versification,
        target_versification: Versification,
    ) -> None:
        self._generator = generator
        self._source_versification = source_versification
        self._is_scripture = (
            source_versification is not None
            and target_versification is not None
            and source_versification != target_versification
        )
        self._is_enumerating = False
        self._verse_rows: SimpleQueue[TextRow] = SimpleQueue()
        self._row: Optional[TextRow] = None

    def send(self, value: None) -> TextRow:
        if self._is_scripture:
            if not self._is_enumerating:
                self._row = next(self._generator, None)
                self._is_enumerating = True
            if self._verse_rows.empty() and self._row is not None:
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
        assert self._source_versification is not None
        seg_list: List[Tuple[ScriptureRef, TextRow]] = []
        out_of_order = False
        prev_scr_ref = EMPTY_SCRIPTURE_REF
        range_start_offset = -1
        while self._row is not None:
            row = self._row
            scr_ref: ScriptureRef = row.ref
            if not prev_scr_ref.is_empty and scr_ref.book_num != prev_scr_ref.book_num:
                break

            scr_ref = scr_ref.change_versification(self._source_versification)
            # convert one-to-many mapping to a verse range
            if scr_ref == prev_scr_ref:
                range_start_verse_ref, range_start_row = seg_list[range_start_offset]
                flags = TextRowFlags.IN_RANGE
                if range_start_row.is_sentence_start:
                    flags |= TextRowFlags.SENTENCE_START
                if range_start_offset == -1 and (not range_start_row.is_in_range or range_start_row.is_range_start):
                    flags |= TextRowFlags.RANGE_START
                seg_list[range_start_offset] = (
                    range_start_verse_ref,
                    TextRow(
                        range_start_row.text_id,
                        range_start_row.ref,
                        list(range_start_row.segment) + list(row.segment),
                        flags,
                    ),
                )
                row = TextRow(row.text_id, row.ref, flags=TextRowFlags.IN_RANGE)
                range_start_offset -= 1
            else:
                range_start_offset = -1
            seg_list.append((scr_ref, row))
            if not out_of_order and scr_ref < prev_scr_ref:
                out_of_order = True
            prev_scr_ref = scr_ref
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
    if isinstance(source_ref, ScriptureRef) and isinstance(target_ref, ScriptureRef):
        return source_ref.compare_to(target_ref, compare_segments=False)
    return compare(source_ref, target_ref)

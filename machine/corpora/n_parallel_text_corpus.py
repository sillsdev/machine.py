from typing import Callable, Generator, Iterable, List, Optional, Sequence, Set, cast

from ..scripture.verse_ref import Versification

from .text_corpus_enumerator import _TextCorpusEnumerator

from .text_row import TextRow, TextRowFlags

from .n_parallel_text_row import NParallelTextRow

from .scripture_ref import ScriptureRef

from .text_corpus import TextCorpus
from .n_parallel_text_corpus_base import NParallelTextCorpusBase


class _RangeRow:
    refs: List[object] = []
    segment: List[str] = []
    is_sentence_start = False

    @property
    def is_in_range(self):
        return len(self.refs) > 0

    @property
    def is_empty(self):
        return len(self.segment) == 0


class _NRangeInfo:
    def __init__(self, n: int):
        self.n = n
        self.rows: List[_RangeRow] = []
        for _ in range(n):
            self.rows.append(_RangeRow())
        self.text_id = ""
        self.versifications: Optional[List[Versification]] = None
        self.row_ref_comparer = None

    @property
    def is_in_range(self) -> bool:
        return any([row.is_in_range for row in self.rows])

    def add_text_row(self, row: TextRow, index: int):
        self.text_id = row.text_id
        self.rows[index].refs.append(row.ref)
        if self.rows[index].is_empty:
            self.rows[index].is_sentence_start = row.is_sentence_start
        self.rows[index].segment.extend(row.segment)

    def create_row(self) -> NParallelTextRow:
        refs: List[List[object]] = [[] for _ in range(self.n)]
        reference_refs: List[object] = [r.refs[0] if len(r.refs) > 0 else None for r in self.rows if len(r.refs) > 0]
        for i in range(len(self.rows)):
            row = self.rows[i]

            if (
                self.versifications is not None
                and all([v is not None for v in self.versifications])
                and len(row.refs) == 0
            ):
                refs[i] = [cast(ScriptureRef, r).change_versification(self.versifications[i]) for r in reference_refs]
            else:
                refs[i] = row.refs
        n_parallel_text_row = NParallelTextRow(self.text_id, refs)
        n_parallel_text_row.n_segments = [r.segment for r in self.rows]
        n_parallel_text_row.n_flags = [
            TextRowFlags.SENTENCE_START if r.is_sentence_start else TextRowFlags.NONE for r in self.rows
        ]
        self.text_id = ""
        for row in self.rows:
            row.refs.clear()
            row.segment.clear()
            row.is_sentence_start = False
        return n_parallel_text_row


class NParallelTextCorpus(NParallelTextCorpusBase):
    def __init__(
        self, corpora: Sequence[TextCorpus], row_ref_comparer: Optional[Callable[[object, object], int]] = None
    ):
        self._corpora = corpora
        self._row_ref_comparer = row_ref_comparer if row_ref_comparer is not None else default_row_ref_comparer
        self.all_rows = [False for _ in range(len(corpora))]

    def is_tokenized(self, i: int) -> bool:
        return self.corpora[i].is_tokenized

    @property
    def n(self) -> int:
        return len(self.corpora)

    @property
    def corpora(self) -> List[TextCorpus]:
        return list(self._corpora)

    @property
    def row_ref_comparer(self) -> Callable[[object, object], int]:
        return self._row_ref_comparer

    def _get_text_ids_from_corpora(self) -> Set[str]:
        text_ids: Set[str] = set()
        all_rows_text_ids: Set[str] = set()
        for i in range(self.n):
            if i == 0:
                text_ids = text_ids.union({text.id for text in self.corpora[i].texts})
            else:
                text_ids = text_ids.intersection({text.id for text in self.corpora[i].texts})

            if self.all_rows[i]:
                all_rows_text_ids = all_rows_text_ids.union({text.id for text in self.corpora[i].texts})
        text_ids = text_ids.union(all_rows_text_ids)
        return text_ids

    def get_rows(self, text_ids: Optional[Sequence[str]]) -> Iterable[NParallelTextRow]:
        filter_text_ids = self._get_text_ids_from_corpora()
        if text_ids is not None:
            filter_text_ids = filter_text_ids.intersection(text_ids)
        enumerated_corpora: List[_TextCorpusEnumerator] = []
        for i in range(self.n):
            enumerator = iter(self.corpora[i].get_rows(filter_text_ids))
            enumerated_corpora.append(
                _TextCorpusEnumerator(enumerator, self.corpora[0].versification, self.corpora[i].versification)
            )
        for row in self._get_rows(enumerated_corpora):
            yield row

    @staticmethod
    def _all_ranges_are_new_ranges(rows: List[TextRow]):
        return all([True if row.is_range_start or not row.is_in_range else False for row in rows])

    def _min_ref_indexes(self, refs: Sequence[object]) -> Sequence[int]:
        min_ref = refs[0]
        min_ref_indexes = [0]
        for i in range(len(refs)):
            if self.row_ref_comparer(refs[i], min_ref) < 0:
                min_ref = refs[i]
                min_ref_indexes = [i]
            elif self.row_ref_comparer(refs[i], min_ref) == 0:
                min_ref_indexes.append(i)
        return min_ref_indexes

    def _get_rows(self, enumerators: List[_TextCorpusEnumerator]) -> Iterable[NParallelTextRow]:
        range_info = _NRangeInfo(self.n)
        same_ref_rows: List[List[TextRow]] = []
        for _ in range(self.n):
            same_ref_rows.append([])

        completed = [False for _ in range(self.n)]
        num_completed = 0
        for i in range(self.n):
            is_completed = not bool(enumerators[i].move_next())
            completed[i] = is_completed
            if is_completed:
                num_completed += 1
        num_remaining_rows = self.n - num_completed

        while num_completed < self.n:
            current_rows = [cast(TextRow, e.current) for e in enumerators]
            refs = []
            for i, row in enumerate(current_rows):
                if row is not None and not completed[i]:
                    refs.append(row.ref)
                else:
                    refs.append(None)
            min_ref_indexes = self._min_ref_indexes(refs)
            non_min_ref_indexes = list(set(range(0, self.n)).difference(min_ref_indexes))
            if len(min_ref_indexes) < num_remaining_rows or len([i for i in min_ref_indexes if not completed[i]]):
                # then there are some non-min refs or only one incomplete enumerator
                if any([not self.all_rows[i] for i in non_min_ref_indexes]) and any(
                    [not completed[i] and current_rows[i].is_in_range for i in min_ref_indexes]
                ):
                    # At least one of the non-min rows has not been marked as 'all rows'
                    # and at least one of the min rows is not completed and in a range
                    for i in min_ref_indexes:
                        range_info.add_text_row(cast(TextRow, enumerators[i].current), i)
                    for i in non_min_ref_indexes:
                        same_ref_rows[i].clear()
                else:
                    any_non_min_enumerators_mid_range = any(
                        [not completed[i] and not current_rows[i].is_range_start and current_rows[i].is_in_range]
                    )
                    for row in self._create_min_ref_rows(
                        range_info,
                        current_rows,
                        min_ref_indexes,
                        non_min_ref_indexes,
                        same_ref_rows,
                        [
                            i in min_ref_indexes
                            and any_non_min_enumerators_mid_range
                            and all(
                                [
                                    not completed[j] and current_rows[j].text_id == current_rows[i].text_id
                                    for j in non_min_ref_indexes
                                ]
                            )
                        ],
                    ):
                        yield row
                for i in min_ref_indexes:
                    if completed[i]:
                        continue
                    same_ref_rows[i].append(cast(TextRow, enumerators[i].current))
                    is_completed = not enumerators[i].move_next()
                    completed[i] = is_completed
                    if is_completed:
                        num_completed += 1
                        num_remaining_rows -= 1

            elif len(min_ref_indexes) == num_remaining_rows:
                # the refs are all the same
                if any(
                    [
                        current_rows[i].is_in_range and all([j == i or not self.all_rows[j] for j in min_ref_indexes])
                        for i in min_ref_indexes
                    ]
                ):
                    # At least one row is in range while the other rows are all not marked as 'all rows'
                    if range_info.is_in_range and NParallelTextCorpus._all_ranges_are_new_ranges(
                        [row for (i, row) in enumerate(current_rows) if not completed[i]]
                    ):
                        yield range_info.create_row()

                    for i in range(len(range_info.rows)):
                        if completed[i]:
                            continue
                        range_info.add_text_row(current_rows[i], i)
                        same_ref_rows[i].clear()
                else:
                    for row in self._create_same_ref_rows(range_info, completed, current_rows, same_ref_rows):
                        yield row

                    for row in self._create_rows(
                        range_info, [r if completed[i] else None for (i, r) in enumerate(current_rows)]
                    ):
                        yield row

                    for i in range(len(range_info.rows)):
                        if completed[i]:
                            continue
                        same_ref_rows[i].append(current_rows[i])
                        is_completed = not enumerators[i].move_next()
                        completed[i] = is_completed
                        if is_completed:
                            num_completed += 1
                            num_remaining_rows -= 1

        if range_info.is_in_range:
            yield range_info.create_row()

    def _correct_versification(self, refs: List[object], i: int) -> List[object]:
        if any([not c.is_scripture for c in self.corpora]) or len(refs) == 0:
            return refs
        return [cast(ScriptureRef, ref).change_versification(self.corpora[i].versification) for ref in refs]

    def _create_rows(
        self, range_info: _NRangeInfo, rows: List[Optional[TextRow]], force_in_range: Optional[Sequence[bool]] = None
    ) -> Iterable[NParallelTextRow]:
        if range_info.is_in_range:
            yield range_info.create_row()

        default_refs = [r.ref for r in rows if r is not None][0]
        text_id: Optional[str] = None
        refs: List[List[object]] = []
        flags: List[TextRowFlags] = []
        for i in range(self.n):
            refs.append([])
            flags[i] = TextRowFlags.NONE
        for i in range(len(rows)):
            row = rows[i]
            if row is not None:
                text_id = text_id or row.text_id
                if self.corpora[i].is_scripture:
                    row = self._correct_versification([row.ref] if row.ref is None else default_refs, i)
                else:
                    refs[i] = default_refs
            else:
                if self.corpora[i].is_scripture:
                    refs[i] = self._correct_versification(default_refs, i)
                else:
                    refs[i] = default_refs
                    flags[i] = (
                        TextRowFlags.IN_RANGE if force_in_range is not None and force_in_range[i] else TextRowFlags.NONE
                    )
        refs = [r or default_refs for r in refs]

        new_row = NParallelTextRow(cast(str, text_id), refs)
        new_row.n_segments = [r.segment if r is not None else [] for r in rows]
        new_row.n_flags = flags
        yield new_row

    def _create_min_ref_rows(
        self,
        range_info: _NRangeInfo,
        current_rows: Sequence[TextRow],
        min_ref_indexes: Sequence[int],
        non_min_ref_indexes: Sequence[int],
        same_ref_rows_per_index: Sequence[List[TextRow]],
        force_in_range: Optional[Sequence[bool]],
    ) -> Iterable[NParallelTextRow]:
        already_yielded: Set[int] = set()
        text_rows: List[Optional[TextRow]] = [None for _ in range(self.n)]
        for i in min_ref_indexes:
            text_row = current_rows[i]
            for j in non_min_ref_indexes:
                same_ref_rows = same_ref_rows_per_index[j]
                if self._check_same_ref_rows(same_ref_rows, text_row):
                    already_yielded.add(i)
                    for same_ref_row in same_ref_rows:
                        text_rows[i] = text_row
                        text_rows[j] = same_ref_row
                        for row in self._create_rows(range_info, text_rows, force_in_range):
                            yield row
        text_rows = [None for _ in range(self.n)]
        rows_have_content = False
        for i in min_ref_indexes:
            if not self.all_rows[i] or i in already_yielded:
                continue
            text_row = current_rows[i]
            text_rows[i] = text_row
            rows_have_content = True

        if rows_have_content:
            for row in self._create_rows(range_info, text_rows, force_in_range):
                yield row

    def _check_same_ref_rows(self, same_ref_rows: List[TextRow], other_row: TextRow) -> bool:
        if len(same_ref_rows) > 0 and self.row_ref_comparer(same_ref_rows[0], other_row.ref) != 0:
            same_ref_rows.clear()
        return len(same_ref_rows) > 0

    def _create_same_ref_rows(
        self,
        range_info: _NRangeInfo,
        completed: Sequence[int],
        current_rows: Sequence[TextRow],
        same_ref_rows: Sequence[List[TextRow]],
    ) -> Iterable[NParallelTextRow]:
        for i in range(self.n):
            if completed[i]:
                continue
            for j in range(self.n):
                if i == j or completed[j]:
                    continue

                if self._check_same_ref_rows(same_ref_rows[i], current_rows[j]):
                    for tr in same_ref_rows[i]:
                        text_rows: List[Optional[TextRow]] = [None for _ in range(self.n)]
                        text_rows[i] = tr
                        text_rows[j] = current_rows[j]
                        for r in self._create_rows(range_info, text_rows):
                            yield r


@staticmethod
def default_row_ref_comparer(x: object, y: object) -> int:
    # Do not use the default comparer for ScriptureRef, since we want to ignore segments
    if isinstance(x, ScriptureRef) and isinstance(y, ScriptureRef):
        return x.compare_to(y, False)
    if x is None and y is not None:
        return 1
    if x is not None and y is None:
        return -1
    if x == y:
        return 0
    if x < y:  # type: ignore
        return -1
    return 1

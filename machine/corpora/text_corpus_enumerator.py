from typing import Generator, List, Optional, Tuple, cast, overload

from ..utils.context_managed_generator import ContextManagedGenerator

from .scripture_ref import ScriptureRef

from ..scripture.verse_ref import Versification
from .text_row import TextRow, TextRowFlags


class _TextCorpusEnumerator(ContextManagedGenerator[TextRow, None, None]):
    def __init__(
        self, enumerator: Generator[TextRow, None, None], ref_versification: Versification, versification: Versification
    ):
        self._enumerator = enumerator
        self._ref_versification = ref_versification
        self._is_scripture = (
            ref_versification is not None and versification is not None and ref_versification != versification
        )
        self._verse_rows: List[TextRow] = []
        self._is_enumerating = False
        self._enumerator_has_more_data = True
        self._current: Optional[TextRow] = None

    def __iter__(self):
        return self

    def __next__(self) -> TextRow:
        if not self.move_next() or self._current is None:
            raise StopIteration
        return self._current

    @property
    def current(self):
        return self._current

    def move_next(self) -> bool:
        if self._is_scripture:
            if not self._is_enumerating:
                self._current = self._enumerator.__next__()
                self._is_enumerating = True
            if len(self._verse_rows) == 0 and self._current is not None and self._enumerator_has_more_data:
                self._collect_verses()
            if len(self._verse_rows) > 0:
                self._current = self._verse_rows.pop(0)
                return True
            self._current = None
            return False

        self._current = self._enumerator.__next__()
        self._enumerator_has_more_data = self._current != None
        return self._enumerator_has_more_data

    # Not porting reset() since it is unused

    def _collect_verses(self):
        rows: List[Tuple[ScriptureRef, TextRow]] = []
        out_of_order: bool = False
        prev_ref = ScriptureRef._empty
        range_start_offset: int = -1
        while True:
            row = cast(TextRow, self._current)
            ref = cast(ScriptureRef, row.ref)
            if prev_ref is not None and not prev_ref.is_empty and ref.book_num != prev_ref.book_num:
                break

            ref = ref.change_versification(self._ref_versification)
            if ref == prev_ref:
                range_start_ref, range_start_row = rows[len(rows) + range_start_offset]
                flags = TextRowFlags.IN_RANGE
                if range_start_row.is_sentence_start:
                    flags |= TextRowFlags.SENTENCE_START
                if range_start_offset == -1 and (not range_start_row.is_in_range or range_start_row.is_range_start):
                    flags |= TextRowFlags.RANGE_START
                    new_text_row = TextRow(range_start_row.text_id, range_start_row.ref)
                    new_text_row.segment = list(range_start_row.segment) + list(row.segment)
                    new_text_row.flags = flags
                rows[len(rows) + range_start_offset] = range_start_ref, new_text_row
                range_start_offset -= 1
            else:
                range_start_offset = -1
            rows.append((ref, row))
            if not out_of_order and ref.compare_to(prev_ref) < 0:
                out_of_order = True
            prev_ref = ref
            self._enumerator_has_more_data = bool(self._enumerator.__next__())
            if not self._enumerator_has_more_data:
                break

        if out_of_order:
            rows.sort(key=lambda tup: tup[0])

        for _, row in rows:
            self._verse_rows.append(row)

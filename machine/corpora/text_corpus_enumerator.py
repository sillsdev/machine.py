from queue import SimpleQueue
from typing import Any, ContextManager, Generator, List, Optional, Tuple, cast

from ..scripture.verse_ref import Versification
from .scripture_ref import EMPTY_SCRIPTURE_REF, ScriptureRef
from .text_row import TextRow, TextRowFlags


class TextCorpusEnumerator(ContextManager["TextCorpusEnumerator"], Generator[TextRow, None, None]):
    def __init__(
        self,
        generator: Generator[TextRow, None, None],
        ref_versification: Optional[Versification],
        versification: Optional[Versification],
    ):
        self._generator = generator
        self._ref_versification = ref_versification
        self._is_scripture = (
            ref_versification is not None and versification is not None and ref_versification != versification
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

    def __enter__(self) -> "TextCorpusEnumerator":
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.close()

    def _collect_verses(self):
        assert self._ref_versification is not None
        rows: List[Tuple[ScriptureRef, TextRow]] = []
        out_of_order = False
        prev_ref = EMPTY_SCRIPTURE_REF
        range_start_offset = -1
        while self._row is not None:
            row = cast(TextRow, self._row)
            ref = cast(ScriptureRef, row.ref)
            if not prev_ref.is_empty and ref.book_num != prev_ref.book_num:
                break

            ref = ref.change_versification(self._ref_versification)
            # convert one-to-many mapping to a verse range
            if ref == prev_ref:
                range_start_ref, range_start_row = rows[range_start_offset]
                flags = TextRowFlags.IN_RANGE
                if range_start_row.is_sentence_start:
                    flags |= TextRowFlags.SENTENCE_START
                if range_start_offset == -1 and (not range_start_row.is_in_range or range_start_row.is_range_start):
                    flags |= TextRowFlags.RANGE_START
                new_text_row = TextRow(
                    range_start_row.text_id,
                    range_start_row.ref,
                    segment=list(range_start_row.segment) + list(row.segment),
                    flags=flags,
                )
                rows[range_start_offset] = range_start_ref, new_text_row
                row = TextRow(row.text_id, row.ref, flags=TextRowFlags.IN_RANGE)
                range_start_offset -= 1
            else:
                range_start_offset = -1
            rows.append((ref, row))
            if not out_of_order and ref < prev_ref:
                out_of_order = True
            prev_ref = ref
            self._row = next(self._generator, None)

        if out_of_order:
            rows.sort(key=lambda t: t[0])

        for _, row in rows:
            self._verse_rows.put(row)

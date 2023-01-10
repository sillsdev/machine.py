from typing import Generator, List, Optional

from ..scripture.verse_ref import VerseRef, Versification, VersificationType
from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_utils import gen, get_scripture_text_sort_key
from .text_base import TextBase
from .text_row import TextRow, TextRowFlags


class ScriptureText(TextBase):
    def __init__(self, id: str, versification: Optional[Versification] = None) -> None:
        super().__init__(id, get_scripture_text_sort_key(id))
        self._versification = (
            Versification.get_builtin(VersificationType.ENGLISH) if versification is None else versification
        )

    @property
    def versification(self) -> Versification:
        return self._versification

    def get_rows(self) -> ContextManagedGenerator[TextRow, None, None]:
        seg_list: List[TextRow] = []
        out_of_order = False
        prev_verse_ref = VerseRef()
        with super().get_rows() as rows:
            for row in rows:
                verse_ref: VerseRef = row.ref
                seg_list.append(row)
                if not out_of_order and verse_ref < prev_verse_ref:
                    out_of_order = True
                prev_verse_ref = verse_ref
        if out_of_order:
            seg_list.sort(key=lambda r: r.ref)
        return ContextManagedGenerator(gen(seg_list))

    def _create_rows(
        self, verse_ref: VerseRef, text: str = "", is_sentence_start: bool = True
    ) -> Generator[TextRow, None, None]:
        if verse_ref.has_multiple:
            first_verse = True
            for vref in verse_ref.all_verses():
                if first_verse:
                    flags = TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START
                    if is_sentence_start:
                        flags |= TextRowFlags.SENTENCE_START
                    yield self._create_row(text, vref, flags)
                    first_verse = False
                else:
                    yield self._create_empty_row(vref, TextRowFlags.IN_RANGE)
        else:
            yield self._create_row(
                text, verse_ref, TextRowFlags.SENTENCE_START if is_sentence_start else TextRowFlags.NONE
            )

    def _create_verse_ref(self, chapter: str, verse: str) -> VerseRef:
        return VerseRef(self.id, chapter, verse, self._versification)

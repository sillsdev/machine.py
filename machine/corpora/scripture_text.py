from typing import Generator, List, Optional, Sequence, Union

from ..scripture import ENGLISH_VERSIFICATION
from ..scripture.verse_ref import VerseRef, Versification
from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_utils import gen, get_scripture_text_sort_key
from .scripture_ref import ScriptureElement, ScriptureRef
from .text_base import TextBase
from .text_row import TextRow, TextRowFlags


class ScriptureText(TextBase):
    def __init__(self, id: str, versification: Optional[Versification] = None) -> None:
        super().__init__(id, get_scripture_text_sort_key(id))
        self._versification = ENGLISH_VERSIFICATION if versification is None else versification

    @property
    def versification(self) -> Versification:
        return self._versification

    def get_rows(self) -> ContextManagedGenerator[TextRow, None, None]:
        seg_list: List[TextRow] = []
        out_of_order = False
        prev_scr_ref = ScriptureRef()
        with super().get_rows() as rows:
            for row in rows:
                scr_ref: ScriptureRef = row.ref
                seg_list.append(row)
                if not out_of_order and scr_ref < prev_scr_ref:
                    out_of_order = True
                prev_scr_ref = scr_ref
        if out_of_order:
            seg_list.sort(key=lambda r: r.ref)
        return ContextManagedGenerator(gen(seg_list))

    def _create_scripture_rows(
        self, ref: Union[Sequence[ScriptureRef], VerseRef], text: str = "", is_sentence_start: bool = True
    ) -> Generator[TextRow, None, None]:
        if isinstance(ref, VerseRef):
            yield from self._create_scripture_rows_verse_ref(ref, text, is_sentence_start)
        else:
            yield from self._create_scripture_rows_scripture_ref(ref, text, is_sentence_start)

    def _create_scripture_rows_scripture_ref(
        self, scripture_refs: Sequence[ScriptureRef], text: str = "", is_sentence_start: bool = True
    ) -> Generator[TextRow, None, None]:
        if len(scripture_refs) > 1:
            first_verse = True
            for sref in scripture_refs:
                if first_verse:
                    flags: TextRowFlags = TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START
                    if is_sentence_start:
                        flags |= TextRowFlags.SENTENCE_START
                    yield self._create_row(text, sref, flags)
                    first_verse = False
                else:
                    yield self._create_empty_row(sref, TextRowFlags.IN_RANGE)
        else:
            yield self._create_row(
                text, scripture_refs[0], TextRowFlags.SENTENCE_START if is_sentence_start else TextRowFlags.NONE
            )

    def _create_scripture_rows_verse_ref(
        self, verse_ref: VerseRef, text: str = "", is_sentence_start: bool = True
    ) -> Generator[TextRow, None, None]:
        if verse_ref.has_multiple:
            first_verse = True
            for vref in verse_ref.all_verses():
                if first_verse:
                    flags = TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START
                    if is_sentence_start:
                        flags |= TextRowFlags.SENTENCE_START
                    yield self._create_row(text, ScriptureRef(vref), flags)
                    first_verse = False
                else:
                    yield self._create_empty_row(ScriptureRef(vref), TextRowFlags.IN_RANGE)
        else:
            yield self._create_row(
                text, ScriptureRef(verse_ref), TextRowFlags.SENTENCE_START if is_sentence_start else TextRowFlags.NONE
            )

    def _create_scripture_row(
        self,
        ref: Union[ScriptureRef, VerseRef],
        text: str,
        is_sentence_start: bool,
        elements: Optional[List[ScriptureElement]] = None,
    ) -> TextRow:
        if isinstance(ref, VerseRef):
            return self._create_row(
                text,
                ScriptureRef(ref, elements),
                TextRowFlags.SENTENCE_START if is_sentence_start else TextRowFlags.NONE,
            )
        else:
            return self._create_row(
                text,
                ref,
                TextRowFlags.SENTENCE_START if is_sentence_start else TextRowFlags.NONE,
            )

    def _create_verse_ref(self, chapter: str, verse: str) -> VerseRef:
        return VerseRef(self.id, chapter, verse, self._versification)

from typing import Generator, List, Optional, Tuple

from ..scripture.verse_ref import VerseRef, Versification, VersificationType
from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_helpers import get_scripture_text_sort_key
from .text_base import TextBase
from .text_corpus_row import TextCorpusRow


class ScriptureText(TextBase):
    def __init__(self, id: str, versification: Optional[Versification] = None) -> None:
        super().__init__(id, get_scripture_text_sort_key(id))
        self._versification = (
            Versification.get_builtin(VersificationType.ENGLISH) if versification is None else versification
        )

    @property
    def versification(self) -> Versification:
        return self._versification

    def get_rows(
        self, versification: Optional[Versification] = None
    ) -> ContextManagedGenerator[TextCorpusRow, None, None]:
        seg_list: List[Tuple[VerseRef, TextCorpusRow]] = []
        out_of_order = False
        prev_verse_ref = VerseRef()
        range_start_offset = -1
        with super().get_rows() as rows:
            for row in rows:
                verse_ref: VerseRef
                if versification is None or versification == self.versification:
                    verse_ref = row.ref
                else:
                    verse_ref = row.ref
                    verse_ref = verse_ref.copy()
                    verse_ref.change_versification(versification)
                    # convert one-to-many mapping to a verse range
                    if verse_ref == prev_verse_ref:
                        range_start_verse_ref, range_start_seg = seg_list[range_start_offset]
                        is_range_start = False
                        if range_start_offset == -1:
                            is_range_start = range_start_seg.is_range_start if range_start_seg.is_in_range else True
                        seg_list[range_start_offset] = (
                            range_start_verse_ref,
                            TextCorpusRow(
                                range_start_seg.ref,
                                list(range_start_seg.segment) + list(row.segment),
                                range_start_seg.is_sentence_start,
                                is_in_range=True,
                                is_range_start=is_range_start,
                                is_empty=range_start_seg.is_empty and row.is_empty,
                            ),
                        )
                        row = self._create_empty_row(row.ref, is_in_range=True)
                        range_start_offset -= 1
                    else:
                        range_start_offset = -1
                seg_list.append((verse_ref, row))
                if not out_of_order and verse_ref < prev_verse_ref:
                    out_of_order = True
                prev_verse_ref = verse_ref
        if out_of_order:
            seg_list.sort(key=lambda t: t[0])
        return ContextManagedGenerator(s for _, s in seg_list)

    def _create_rows(
        self, chapter: str, verse: str, text: str, is_sentence_start: bool = True
    ) -> Generator[TextCorpusRow, None, None]:
        verse_ref = VerseRef(self.id, chapter, verse, self._versification)
        if verse_ref.has_multiple:
            first_verse = True
            for vref in verse_ref.all_verses():
                if first_verse:
                    yield self._create_row(text, vref, is_sentence_start, is_in_range=True, is_range_start=True)
                    first_verse = False
                else:
                    yield self._create_empty_row(vref, is_in_range=True)
        else:
            yield self._create_row(text, verse_ref, is_sentence_start)

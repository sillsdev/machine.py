from abc import abstractmethod
from typing import Generator, List, Optional, cast

from ..scripture.verse_ref import VerseRef, Versification, VersificationType
from ..tokenization import Tokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_helpers import gen, get_scripture_text_sort_key
from .text import Text
from .text_base import TextBase
from .text_segment import TextSegment


class ScriptureText(TextBase):
    def __init__(
        self, word_tokenizer: Tokenizer[str, int, str], id: str, versification: Optional[Versification] = None
    ) -> None:
        super().__init__(word_tokenizer, id, get_scripture_text_sort_key(id))
        self._versification = (
            Versification.get_builtin(VersificationType.ENGLISH) if versification is None else versification
        )

    @property
    def versification(self) -> Versification:
        return self._versification

    def _get_segments(self, include_text: bool, based_on: Optional[Text]) -> Generator[TextSegment, None, None]:
        based_on_versification: Optional[Versification] = None
        if isinstance(based_on, ScriptureText) and self.versification != based_on.versification:
            based_on_versification = based_on.versification
        seg_list: List[TextSegment] = []
        out_of_order = False
        with ContextManagedGenerator(self._get_segments_in_doc_order(include_text)) as segs:
            prev_verse_ref = VerseRef()
            for seg in segs:
                if based_on_versification is not None:
                    cast(VerseRef, seg.segment_ref).change_versification(based_on_versification)
                seg_list.append(seg)
                if seg.segment_ref < prev_verse_ref:
                    out_of_order = True
                prev_verse_ref = seg.segment_ref
        if out_of_order:
            seg_list.sort(key=lambda s: s.segment_ref)
        return ContextManagedGenerator(gen(seg_list))

    @abstractmethod
    def _get_segments_in_doc_order(self, include_text: bool) -> Generator[TextSegment, None, None]:
        ...

    def _create_text_segments(
        self, include_text: bool, chapter: str, verse: str, text: str, is_sentence_start: bool = True
    ) -> Generator[TextSegment, None, None]:
        verse_ref = VerseRef(self.id, chapter, verse, self._versification)
        if verse_ref.has_multiple:
            first_verse = True
            for vref in verse_ref.all_verses():
                if first_verse:
                    yield self._create_text_segment(
                        include_text, text, vref, is_sentence_start, is_in_range=True, is_range_start=True
                    )
                    first_verse = False
                else:
                    yield self._create_empty_text_segment(vref, is_in_range=True)
        else:
            yield self._create_text_segment(include_text, text, verse_ref, is_sentence_start)

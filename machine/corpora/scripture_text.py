from typing import Iterable, List, Optional, cast

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

    def get_segments_based_on(
        self, text: Text, include_text: bool = True
    ) -> ContextManagedGenerator[TextSegment, None, None]:
        if not isinstance(text, ScriptureText) or self.versification == text.versification:
            return super().get_segments_based_on(text, include_text)

        segments: List[TextSegment] = []
        with self.get_segments(include_text) as segs:
            for seg in segs:
                cast(VerseRef, seg.segment_ref).change_versification(text.versification)
                segments.append(seg)

        segments.sort(key=lambda s: s.segment_ref)
        return ContextManagedGenerator(gen(segments))

    def _create_text_segments(
        self,
        include_text: bool,
        prev_verse_ref: VerseRef,
        chapter: str,
        verse: str,
        text: str,
        is_sentence_start: bool = True,
    ) -> Iterable[TextSegment]:
        verse_ref = VerseRef(self.id, chapter, verse, self._versification)
        if verse_ref <= prev_verse_ref:
            return []

        segments: List[TextSegment] = []
        if verse_ref.has_multiple:
            first_verse = True
            for vref in verse_ref.all_verses():
                if first_verse:
                    segments.append(
                        self._create_text_segment(
                            include_text, text, vref, is_sentence_start, is_in_range=True, is_range_start=True
                        )
                    )
                    first_verse = False
                else:
                    segments.append(self._create_empty_text_segment(vref, is_in_range=True))
        else:
            segments.append(self._create_text_segment(include_text, text, verse_ref, is_sentence_start))

        prev_verse_ref.copy_from(verse_ref)

        return segments

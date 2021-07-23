from dataclasses import dataclass
from typing import Any, Collection

from .aligned_word_pair import AlignedWordPair


@dataclass(eq=False, frozen=True)
class TextAlignment:
    text_id: str
    segment_ref: Any
    aligned_word_pairs: Collection[AlignedWordPair]

    def invert(self) -> "TextAlignment":
        return TextAlignment(self.text_id, self.segment_ref, {ta.invert() for ta in self.aligned_word_pairs})

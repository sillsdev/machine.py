from dataclasses import dataclass
from typing import Any, Iterable, Set

from .aligned_word_pair import AlignedWordPair


@dataclass(eq=False, frozen=True)
class TextAlignment:
    segment_ref: Any
    aligned_word_pairs: Set[AlignedWordPair]

    def __init__(self, segment_ref: Any, aligned_word_pairs: Iterable[AlignedWordPair]) -> None:
        object.__setattr__(self, "segment_ref", segment_ref)
        object.__setattr__(self, "aligned_word_pairs", set(aligned_word_pairs))

    def invert(self) -> "TextAlignment":
        return TextAlignment(self.segment_ref, ta.invert() for ta in self.aligned_word_pairs)



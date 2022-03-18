from dataclasses import dataclass
from typing import Any, Collection

from .aligned_word_pair import AlignedWordPair


@dataclass
class TextAlignmentCorpusRow:
    ref: Any
    aligned_word_pairs: Collection[AlignedWordPair]

    def invert(self) -> "TextAlignmentCorpusRow":
        return TextAlignmentCorpusRow(self.ref, {ta.invert() for ta in self.aligned_word_pairs})

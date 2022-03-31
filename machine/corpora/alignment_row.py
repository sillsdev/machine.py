from typing import Any, Collection

from .aligned_word_pair import AlignedWordPair


class AlignmentRow:
    def __init__(self, ref: Any, aligned_word_pairs: Collection[AlignedWordPair] = []) -> None:
        self._ref = ref
        self.aligned_word_pairs = aligned_word_pairs

    @property
    def ref(self) -> Any:
        return self._ref

    @property
    def is_empty(self) -> bool:
        return len(self.aligned_word_pairs) == 0

    def invert(self) -> "AlignmentRow":
        return AlignmentRow(self.ref, {ta.invert() for ta in self.aligned_word_pairs})

from typing import Any, Collection

from .aligned_word_pair import AlignedWordPair


class AlignmentRow:
    def __init__(self, text_id: str, ref: Any, aligned_word_pairs: Collection[AlignedWordPair] = []) -> None:
        self._text_id = text_id
        self._ref = ref
        self.aligned_word_pairs = aligned_word_pairs

    @property
    def text_id(self) -> str:
        return self._text_id

    @property
    def ref(self) -> Any:
        return self._ref

    def invert(self) -> "AlignmentRow":
        return AlignmentRow(self.ref, {ta.invert() for ta in self.aligned_word_pairs})
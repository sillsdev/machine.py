from typing import List, Sequence

from .text_row import TextRowFlags


class NParallelTextRow:
    def __init__(self, text_id: str, n_refs: Sequence[Sequence[object]]):
        if len([n_ref for n_ref in n_refs if n_ref is not None]) == 0:
            raise ValueError(f"Refs must be provided but n_refs={n_refs}")
        self._text_id = text_id
        self._n_refs = n_refs
        self._n = len(n_refs)
        self.n_segments: Sequence[Sequence[str]] = [[] for _ in range(0, self._n)]
        self.n_flags: Sequence[TextRowFlags] = [TextRowFlags.SENTENCE_START for _ in range(0, self._n)]

    @property
    def text_id(self) -> str:
        return self._text_id

    @property
    def ref(self) -> object:
        return self._n_refs[0][0]

    @property
    def n_refs(self) -> Sequence[Sequence[object]]:
        return self._n_refs

    def is_sentence_start(self, i: int) -> bool:
        return TextRowFlags.SENTENCE_START in self.n_flags[i]

    def is_in_range(self, i: int) -> bool:
        return TextRowFlags.IN_RANGE in self.n_flags[i]

    def is_range_start(self, i: int) -> bool:
        return TextRowFlags.RANGE_START in self.n_flags[i]

    @property
    def is_empty(self):
        return sum([1 for s in self.n_segments if len(s) == 0]) == 0

    def text(self, i: int) -> str:
        return " ".join(self.n_segments[i])

    def invert(self) -> "NParallelTextRow":
        inverted_row = NParallelTextRow(self._text_id, list(reversed(self._n_refs)))
        inverted_row.n_flags = list(reversed(self.n_flags))
        return inverted_row

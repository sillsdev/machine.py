from __future__ import annotations

from typing import Any, Collection, Optional, Sequence

from .aligned_word_pair import AlignedWordPair
from .text_row import TextRowFlags


class ParallelTextRow(Sequence[Sequence[str]]):
    def __init__(
        self,
        text_id: str,
        source_refs: Sequence[Any],
        target_refs: Sequence[Any],
        source_segment: Sequence[str] = [],
        target_segment: Sequence[str] = [],
        aligned_word_pairs: Optional[Collection[AlignedWordPair]] = None,
        source_flags: TextRowFlags = TextRowFlags.SENTENCE_START,
        target_flags: TextRowFlags = TextRowFlags.SENTENCE_START,
    ) -> None:
        if len(source_refs) == 0 and len(target_refs) == 0:
            raise ValueError("Either a source or target ref must be set.")
        self._text_id = text_id
        self._source_refs = source_refs
        self._target_refs = target_refs
        self.source_segment = source_segment
        self.target_segment = target_segment
        self.aligned_word_pairs = aligned_word_pairs
        self.source_flags = source_flags
        self.target_flags = target_flags

    @property
    def text_id(self) -> str:
        return self._text_id

    @property
    def source_refs(self) -> Sequence[Any]:
        return self._source_refs

    @property
    def target_refs(self) -> Sequence[Any]:
        return self._target_refs

    @property
    def ref(self) -> Any:
        return self.target_refs[0] if len(self.source_refs) == 0 else self.source_refs[0]

    @property
    def refs(self) -> Sequence[Any]:
        return self.target_refs if len(self.source_refs) == 0 else self.source_refs

    @property
    def is_source_sentence_start(self) -> bool:
        return TextRowFlags.SENTENCE_START in self.source_flags

    @property
    def is_source_in_range(self) -> bool:
        return TextRowFlags.IN_RANGE in self.source_flags

    @property
    def is_source_range_start(self) -> bool:
        return TextRowFlags.RANGE_START in self.source_flags

    @property
    def is_target_sentence_start(self) -> bool:
        return TextRowFlags.SENTENCE_START in self.target_flags

    @property
    def is_target_in_range(self) -> bool:
        return TextRowFlags.IN_RANGE in self.target_flags

    @property
    def is_target_range_start(self) -> bool:
        return TextRowFlags.RANGE_START in self.target_flags

    @property
    def is_empty(self) -> bool:
        return len(self.source_segment) == 0 or len(self.target_segment) == 0

    @property
    def source_text(self) -> str:
        return " ".join(self.source_segment)

    @property
    def target_text(self) -> str:
        return " ".join(self.target_segment)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int) -> Sequence[str]:
        if i >= 2:
            raise IndexError
        if i == 0:
            return self.source_segment
        return self.target_segment

    def invert(self) -> ParallelTextRow:
        return ParallelTextRow(
            self.text_id,
            self.target_refs,
            self.source_refs,
            self.target_segment,
            self.source_segment,
            None if self.aligned_word_pairs is None else [wp.invert() for wp in self.aligned_word_pairs],
            self.target_flags,
            self.source_flags,
        )

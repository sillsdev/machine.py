from __future__ import annotations

from typing import Any, Collection, Optional, Sequence

from .aligned_word_pair import AlignedWordPair


class ParallelTextRow:
    def __init__(
        self,
        text_id: str,
        source_refs: Sequence[Any],
        target_refs: Sequence[Any],
        source_segment: Sequence[str] = [],
        target_segment: Sequence[str] = [],
        aligned_word_pairs: Optional[Collection[AlignedWordPair]] = None,
        is_source_sentence_start: bool = True,
        is_source_in_range: bool = False,
        is_source_range_start: bool = False,
        is_target_sentence_start: bool = True,
        is_target_in_range: bool = False,
        is_target_range_start: bool = False,
        is_empty: bool = True,
    ) -> None:
        if len(source_refs) == 0 and len(target_refs) == 0:
            raise ValueError("Either a source or target ref must be set.")
        self._text_id = text_id
        self._source_refs = source_refs
        self._target_refs = target_refs
        self.source_segment = source_segment
        self.target_segment = target_segment
        self.aligned_word_pairs = aligned_word_pairs
        self.is_source_sentence_start = is_source_sentence_start
        self.is_source_in_range = is_source_in_range
        self.is_source_range_start = is_source_range_start
        self.is_target_sentence_start = is_target_sentence_start
        self.is_target_in_range = is_target_in_range
        self.is_target_range_start = is_target_range_start
        self.is_empty = is_empty

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
    def source_text(self) -> str:
        return " ".join(self.source_segment)

    @property
    def target_text(self) -> str:
        return " ".join(self.target_segment)

    def invert(self) -> ParallelTextRow:
        return ParallelTextRow(
            self.text_id,
            self.target_refs,
            self.source_refs,
            self.target_segment,
            self.source_segment,
            None if self.aligned_word_pairs is None else [wp.invert() for wp in self.aligned_word_pairs],
            self.is_target_sentence_start,
            self.is_target_in_range,
            self.is_target_range_start,
            self.is_source_sentence_start,
            self.is_source_in_range,
            self.is_source_range_start,
            self.is_empty,
        )

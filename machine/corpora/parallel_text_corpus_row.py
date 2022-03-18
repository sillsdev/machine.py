from dataclasses import dataclass
from typing import Any, Collection, Optional, Sequence

from .aligned_word_pair import AlignedWordPair


@dataclass
class ParallelTextCorpusRow:
    source_refs: Sequence[Any]
    target_refs: Sequence[Any]
    source_segment: Sequence[str]
    target_segment: Sequence[str]
    aligned_word_pairs: Optional[Collection[AlignedWordPair]]
    is_source_sentence_start: bool
    is_source_in_range: bool
    is_source_range_start: bool
    is_target_sentence_start: bool
    is_target_in_range: bool
    is_target_range_start: bool
    is_empty: bool

    def __post_init__(self) -> None:
        if len(self.source_refs) == 0 and len(self.target_refs) == 0:
            raise ValueError("Either a source or target ref must be set.")

    @property
    def ref(self) -> Any:
        return self.target_refs[0] if len(self.source_refs) == 0 else self.source_refs[0]

    @property
    def source_text(self) -> str:
        return " ".join(self.source_segment)

    @property
    def target_text(self) -> str:
        return " ".join(self.target_segment)

    def invert(self) -> "ParallelTextCorpusRow":
        return ParallelTextCorpusRow(
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

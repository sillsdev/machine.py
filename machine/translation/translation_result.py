from dataclasses import dataclass
from typing import Sequence

from .phrase import Phrase
from .translation_sources import TranslationSources
from .word_alignment_matrix import WordAlignmentMatrix


@dataclass(frozen=True)
class TranslationResult:
    source_segment_length: int
    target_segment: Sequence[str]
    word_confidences: Sequence[float]
    word_sources: Sequence[TranslationSources]
    alignment: WordAlignmentMatrix
    phrases: Sequence[Phrase]

    def __post_init__(self) -> None:
        if len(self.word_confidences) != len(self.target_segment):
            raise ValueError("The confidences must the same length as the target segment.")
        if len(self.word_sources) != len(self.target_segment):
            raise ValueError("The sources must the same length as the target segment.")
        if self.alignment.row_count != self.source_segment_length:
            raise ValueError("The alignment source length must be the same length as the source segment.")
        if self.alignment.column_count != len(self.target_segment):
            raise ValueError("The alignment target length must be the same length as the target segment.")

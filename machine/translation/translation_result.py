from typing import Iterable, Sequence

from .phrase import Phrase
from .translation_sources import TranslationSources
from .word_alignment_matrix import WordAlignmentMatrix


class TranslationResult:
    def __init__(
        self,
        source_segment_length: int,
        target_segment: Iterable[str],
        word_confidences: Iterable[float],
        word_sources: Iterable[TranslationSources],
        alignment: WordAlignmentMatrix,
        phrases: Iterable[Phrase],
    ) -> None:
        self._source_segment_length = source_segment_length
        self._target_segment = list(target_segment)
        self._word_confidences = list(word_confidences)
        self._word_sources = list(word_sources)
        self._alignment = alignment
        self._phrases = list(phrases)

        if len(self._word_confidences) != len(self._target_segment):
            raise ValueError("The confidences must the same length as the target segment.")
        if len(self._word_sources) != len(self._target_segment):
            raise ValueError("The sources must the same length as the target segment.")
        if self._alignment.row_count != self._source_segment_length:
            raise ValueError("The alignment source length must be the same length as the source segment.")
        if self._alignment.column_count != len(self._target_segment):
            raise ValueError("The alignment target length must be the same length as the target segment.")

    @property
    def source_segment_length(self) -> int:
        return self._source_segment_length

    @property
    def target_segment(self) -> Sequence[str]:
        return self._target_segment

    @property
    def word_confidences(self) -> Sequence[float]:
        return self._word_confidences

    @property
    def word_sources(self) -> Sequence[TranslationSources]:
        return self._word_sources

    @property
    def alignment(self) -> WordAlignmentMatrix:
        return self._alignment

    @property
    def phrases(self) -> Sequence[Phrase]:
        return self._phrases

from typing import Iterable, Sequence

from ..annotations import Range
from .translation_sources import TranslationSources
from .word_alignment_matrix import WordAlignmentMatrix


class WordGraphArc:
    def __init__(
        self,
        prev_state: int,
        next_state: int,
        score: float,
        words: Iterable[str],
        alignment: WordAlignmentMatrix,
        source_segment_range: Range[int],
        sources: Iterable[TranslationSources],
        confidences: Iterable[float],
    ) -> None:
        self._prev_state = prev_state
        self._next_state = next_state
        self._score = score
        self._words = list(words)
        self._alignment = alignment
        self._source_segment_range = source_segment_range
        self._sources = list(sources)
        self._confidences = list(confidences)

    @property
    def prev_state(self) -> int:
        return self._prev_state

    @property
    def next_state(self) -> int:
        return self._next_state

    @property
    def score(self) -> float:
        return self._score

    @property
    def words(self) -> Sequence[str]:
        return self._words

    @property
    def alignment(self) -> WordAlignmentMatrix:
        return self._alignment

    @property
    def source_segment_range(self) -> Range[int]:
        return self._source_segment_range

    @property
    def sources(self) -> Sequence[TranslationSources]:
        return self._sources

    @property
    def confidences(self) -> Sequence[float]:
        return self._confidences

    @property
    def is_unknown(self) -> bool:
        return all(s == TranslationSources.NONE for s in self._sources)

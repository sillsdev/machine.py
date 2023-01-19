from itertools import repeat
from typing import Iterable, List, Optional, Sequence

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
        word_sources: Iterable[TranslationSources],
        word_confidences: Optional[Iterable[float]] = None,
    ) -> None:
        self._prev_state = prev_state
        self._next_state = next_state
        self._score = score
        self._words = list(words)
        self._alignment = alignment
        self._source_segment_range = source_segment_range
        self._word_sources = list(word_sources)
        if word_confidences is None:
            word_confidences = repeat(-1, len(self._words))
        self._word_confidences = list(word_confidences)

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
    def word_sources(self) -> Sequence[TranslationSources]:
        return self._word_sources

    @property
    def word_confidences(self) -> List[float]:
        return self._word_confidences

    @property
    def is_unknown(self) -> bool:
        return all(s == TranslationSources.NONE for s in self._word_sources)

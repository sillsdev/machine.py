from typing import Iterable, Sequence

from .phrase import Phrase
from .translation_sources import TranslationSources
from .word_alignment_matrix import WordAlignmentMatrix


class TranslationResult:
    def __init__(
        self,
        translation: str,
        source_tokens: Iterable[str],
        target_tokens: Iterable[str],
        confidences: Iterable[float],
        sources: Iterable[TranslationSources],
        alignment: WordAlignmentMatrix,
        phrases: Iterable[Phrase],
    ) -> None:
        self._translation = translation
        self._source_tokens = list(source_tokens)
        self._target_tokens = list(target_tokens)
        self._confidences = list(confidences)
        self._sources = list(sources)
        self._alignment = alignment
        self._phrases = list(phrases)

        if len(self._confidences) != len(self._target_tokens):
            raise ValueError("The confidences must the same length as the target segment.")
        if len(self._sources) != len(self._target_tokens):
            raise ValueError("The sources must the same length as the target segment.")
        if self._alignment.row_count != len(self._source_tokens):
            raise ValueError("The alignment source length must be the same length as the source segment.")
        if self._alignment.column_count != len(self._target_tokens):
            raise ValueError("The alignment target length must be the same length as the target segment.")

    @property
    def translation(self) -> str:
        return self._translation

    @property
    def source_tokens(self) -> Sequence[str]:
        return self._source_tokens

    @property
    def target_tokens(self) -> Sequence[str]:
        return self._target_tokens

    @property
    def confidences(self) -> Sequence[float]:
        return self._confidences

    @property
    def sources(self) -> Sequence[TranslationSources]:
        return self._sources

    @property
    def alignment(self) -> WordAlignmentMatrix:
        return self._alignment

    @property
    def phrases(self) -> Sequence[Phrase]:
        return self._phrases

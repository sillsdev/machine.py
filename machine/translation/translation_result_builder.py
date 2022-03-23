import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..annotations.range import Range
from .edit_operation import EditOperation
from .phrase import Phrase
from .translation_result import TranslationResult
from .translation_sources import TranslationSources
from .word_alignment_matrix import WordAlignmentMatrix


@dataclass
class PhraseInfo:
    source_segment_range: Range[int]
    target_cut: int
    alignment: WordAlignmentMatrix


class TranslationResultBuilder:
    def __init__(self) -> None:
        self._words: List[str] = []
        self._confidences: List[float] = []
        self._sources: List[TranslationSources] = []
        self._phrases: List[PhraseInfo] = []

    @property
    def words(self) -> Sequence[str]:
        return self._words

    @property
    def confidences(self) -> Sequence[float]:
        return self._confidences

    @property
    def sources(self) -> Sequence[TranslationSources]:
        return self._sources

    @property
    def phrases(self) -> Sequence[PhraseInfo]:
        return self._phrases

    def append_word(self, word: str, source: TranslationSources, confidence: float = -1) -> None:
        self._words.append(word)
        self._sources.append(source)
        self._confidences.append(confidence)

    def mark_phrase(self, source_segment_range: Range[int], alignment: WordAlignmentMatrix) -> None:
        self._phrases.append(PhraseInfo(source_segment_range, len(self._words), alignment))

    def set_confidence(self, index: int, confidence: float) -> None:
        self._confidences[index] = confidence

    def correct_prefix(
        self,
        word_ops: Iterable[EditOperation],
        char_ops: Iterable[EditOperation],
        prefix: List[str],
        is_last_word_complete: bool,
    ) -> int:
        alignment_cols_to_copy: List[int] = []

        i = 0
        j = 0
        k = 0
        for word_op in word_ops:
            if word_op == EditOperation.INSERT:
                self._words.insert(j, prefix[i])
                self._sources.insert(j, TranslationSources.PREFIX)
                self._confidences.insert(j, -1)
                alignment_cols_to_copy.append(-1)
                for pi in range(k, len(self._phrases)):
                    self._phrases[pi].target_cut += 1
                j += 1
            elif word_op == EditOperation.DELETE:
                self._words.pop(j)
                self._sources.pop(j)
                self._confidences.pop(j)
                i += 1
                if k < len(self._phrases):
                    for pi in range(k, len(self._phrases)):
                        self._phrases[pi].target_cut -= 1

                    if self._phrases[k].target_cut <= 0 or (
                        k > 0 and self._phrases[k].target_cut == self._phrases[k - 1].target_cut
                    ):
                        self._phrases.pop(k)
                        alignment_cols_to_copy.clear()
                        i = 0
                    elif j >= self._phrases[k].target_cut:
                        self._resize_alignment(k, alignment_cols_to_copy)
                        alignment_cols_to_copy.clear()
                        i = 0
                        k += 1
            elif word_op in {EditOperation.HIT, EditOperation.SUBSTITUTE}:
                if word_op == EditOperation.SUBSTITUTE or j < len(prefix) - 1 or is_last_word_complete:
                    self._words[j] = prefix[i]
                else:
                    self._words[j] = self._correct_word(char_ops, self._words[j], prefix[i])

                if word_op == EditOperation.SUBSTITUTE:
                    self._confidences[j] = -1
                    self._sources[j] = TranslationSources.PREFIX
                elif word_op == EditOperation.HIT:
                    self._sources[j] |= TranslationSources.PREFIX

                alignment_cols_to_copy.append(i)

                i += 1
                j += 1
                if k < len(self._phrases) and j >= self._phrases[k].target_cut:
                    self._resize_alignment(k, alignment_cols_to_copy)
                    alignment_cols_to_copy.clear()
                    i = 0
                    k += 1

        while j < len(self._words):
            alignment_cols_to_copy.append(i)

            i += 1
            j += 1
            if k < len(self._phrases) and j >= self._phrases[k].target_cut:
                self._resize_alignment(k, alignment_cols_to_copy)
                alignment_cols_to_copy.clear()
                break

        return len(alignment_cols_to_copy)

    def to_result(self, source_segment: Sequence[str]) -> TranslationResult:
        sources = [TranslationSources.NONE] * len(self._words)
        alignment = WordAlignmentMatrix.from_word_pairs(len(source_segment), len(self._words))
        phrases: List[Phrase] = []
        trg_phrase_start_index = 0
        for phrase_info in self._phrases:
            confidence = sys.float_info.max
            for j in range(trg_phrase_start_index, phrase_info.target_cut):
                for i in range(phrase_info.source_segment_range.start, phrase_info.source_segment_range.end):
                    aligned = phrase_info.alignment[
                        i - phrase_info.source_segment_range.start, j - trg_phrase_start_index
                    ]
                    if aligned:
                        alignment[i, j] = True

                sources[j] = self._sources[j]
                confidence = min(confidence, self._confidences[j])

            phrases.append(Phrase(phrase_info.source_segment_range, phrase_info.target_cut, confidence))
            trg_phrase_start_index = phrase_info.target_cut

        return TranslationResult(
            len(source_segment), self._words.copy(), self._confidences.copy(), sources, alignment, phrases
        )

    def _resize_alignment(self, phrase_index: int, cols_to_copy: List[int]) -> None:
        cur_alignment = self._phrases[phrase_index].alignment
        if len(cols_to_copy) == cur_alignment.column_count:
            return

        new_alignment = WordAlignmentMatrix.from_word_pairs(cur_alignment.row_count, len(cols_to_copy))
        for j in range(new_alignment.column_count):
            if cols_to_copy[j] != -1:
                for i in range(new_alignment.row_count):
                    new_alignment[i, j] = cur_alignment[i, cols_to_copy[j]]

        self._phrases[phrase_index].alignment = new_alignment

    def _correct_word(self, char_ops: Iterable[EditOperation], word: str, prefix: str) -> str:
        result = ""
        i = 0
        j = 0
        for char_op in char_ops:
            if char_op == EditOperation.HIT:
                result += word[i]
                i += 1
                j += 1
            elif char_op == EditOperation.INSERT:
                result += prefix[j]
                j += 1
            elif char_op == EditOperation.DELETE:
                i += 1
            elif char_op == EditOperation.SUBSTITUTE:
                result += prefix[j]
                i += 1
                j += 1

        result += word[i:]
        return result

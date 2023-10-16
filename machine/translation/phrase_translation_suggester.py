from typing import Iterable, List, Optional, Sequence

from ..utils.string_utils import is_punctuation
from .translation_result import TranslationResult
from .translation_sources import TranslationSources
from .translation_suggester import TranslationSuggester
from .translation_suggestion import TranslationSuggestion


class PhraseTranslationSuggester(TranslationSuggester):
    def get_suggestions(
        self, n: int, prefix_count: int, is_last_word_complete: bool, results: Iterable[TranslationResult]
    ) -> Sequence[TranslationSuggestion]:
        suggestions: List[TranslationSuggestion] = []
        for result in results:
            starting_j = prefix_count
            if not is_last_word_complete:
                # if the prefix ends with a partial word and it has been completed,
                # then make sure it is included as a suggestion,
                # otherwise, don't return any suggestions
                if TranslationSources.SMT in result.sources[starting_j - 1]:
                    starting_j -= 1
                else:
                    break

            k = 0
            while k < len(result.phrases) and result.phrases[k].target_segment_cut <= starting_j:
                k += 1

            suggestion_confidence = -1
            indices: List[int] = []
            for k in range(k, len(result.phrases)):
                phrase = result.phrases[k]
                phrase_confidence = 1.0
                ending_j = starting_j
                for j in range(starting_j, phrase.target_segment_cut):
                    if result.sources[j] == TranslationSources.NONE:
                        phrase_confidence = 0.0
                        break

                    word = result.target_tokens[j]
                    if self.break_on_punctuation and all(is_punctuation(c) for c in word):
                        break

                    phrase_confidence = min(phrase_confidence, result.confidences[j])
                    if phrase_confidence < self.confidence_threshold:
                        break

                    ending_j = j + 1

                if phrase_confidence >= self.confidence_threshold:
                    suggestion_confidence = (
                        phrase_confidence
                        if suggestion_confidence == -1
                        else min(suggestion_confidence, phrase_confidence)
                    )

                    if starting_j == ending_j:
                        break

                    for j in range(starting_j, ending_j):
                        indices.append(j)

                    starting_j = phrase.target_segment_cut
                else:
                    # hit a phrase with a low confidence, so don't include any more words in this suggestion
                    break
            if suggestion_confidence == -1:
                break
            elif len(indices) == 0:
                continue

            new_suggestion = TranslationSuggestion(result, indices, suggestion_confidence)
            duplicate = False
            new_suggestion_words: Optional[List[str]] = None
            table: Optional[List[int]] = None
            for suggestion in suggestions:
                if len(suggestion.target_word_indices) >= len(new_suggestion.target_word_indices):
                    if new_suggestion_words is None:
                        new_suggestion_words = list(new_suggestion.target_words)
                    if table is None:
                        table = _compute_kmp_table(new_suggestion_words)
                    if _is_subsequence(table, new_suggestion_words, list(suggestion.target_words)):
                        duplicate = True
                        break

            if not duplicate:
                suggestions.append(new_suggestion)
                if len(suggestions) == n:
                    break
        return suggestions


def _is_subsequence(table: List[int], new_suggestion: Sequence[str], suggestion: Sequence[str]) -> bool:
    j = 0
    i = 0
    while i < len(suggestion):
        if new_suggestion[j] == suggestion[i]:
            j += 1
            i += 1
        if j == len(new_suggestion):
            return True
        elif i < len(suggestion) and new_suggestion[j] != suggestion[i]:
            if j != 0:
                j = table[j - 1]
            else:
                i += 1
    return False


def _compute_kmp_table(new_suggestion: Sequence[str]) -> List[int]:
    table = [0] * len(new_suggestion)
    length = 0
    i = 1
    table[0] = 0

    while i < len(new_suggestion):
        if new_suggestion[i] == new_suggestion[length]:
            length += 1
            table[i] = length
            i += 1
        elif length != 0:
            length = table[length - 1]
        else:
            table[i] = length
            i += 1
    return table

from typing import List

from machine.annotations import Range
from machine.translation import (
    PhraseTranslationSuggester,
    TranslationResult,
    TranslationResultBuilder,
    TranslationSources,
    WordAlignmentMatrix,
)


def test_get_suggestions_punctuation() -> None:
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(
        n=1, prefix_count=0, is_last_word_complete=True, results=[builder.to_result()]
    )
    assert list(suggestions[0].target_words) == ["this", "is", "a", "test"]


def test_get_suggestions_untranslated_word() -> None:
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 2),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    builder.append_token("a", TranslationSources.NONE, 0)
    builder.mark_phrase(
        Range.create(2, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=1, column_count=1, set_values=[(0, 0)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(
        n=1, prefix_count=0, is_last_word_complete=True, results=[builder.to_result()]
    )
    assert list(suggestions[0].target_words) == ["this", "is"]


def test_get_suggestions_prefix_incomplete_word() -> None:
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token("this", TranslationSources.SMT | TranslationSources.PREFIX, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(
        n=1, prefix_count=1, is_last_word_complete=False, results=[builder.to_result()]
    )
    assert list(suggestions[0].target_words) == ["this", "is", "a", "test"]


def test_get_suggestions_prefix_complete_word() -> None:
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token("this", TranslationSources.SMT | TranslationSources.PREFIX, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(
        n=1, prefix_count=1, is_last_word_complete=True, results=[builder.to_result()]
    )
    assert list(suggestions[0].target_words) == ["is", "a", "test"]


def test_get_suggestions_prefix_partial_word() -> None:
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token("te", TranslationSources.PREFIX, -1)
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=4, set_values=[(0, 1), (1, 2), (2, 3)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(
        n=1, prefix_count=1, is_last_word_complete=False, results=[builder.to_result()]
    )
    assert suggestions == []


def test_get_suggestions_multiple() -> None:
    results: List[TranslationResult] = []
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    results.append(builder.to_result())

    builder.reset()
    builder.append_token("that", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    results.append(builder.to_result())

    builder.reset()
    builder.append_token("other", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    results.append(builder.to_result())

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(n=2, prefix_count=0, is_last_word_complete=True, results=results)
    assert len(suggestions) == 2
    assert list(suggestions[0].target_words) == ["this", "is", "a", "test"]
    assert list(suggestions[1].target_words) == ["that", "is", "a", "test"]


def test_get_suggestions_duplicate() -> None:
    results: List[TranslationResult] = []
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", ".", "segunda", "frase"])
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    builder.append_token("second", TranslationSources.SMT, 0.1)
    builder.append_token("sentence", TranslationSources.SMT, 0.1)
    builder.mark_phrase(
        Range.create(5, 7),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    results.append(builder.to_result())

    builder.reset()
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=2, set_values=[(1, 0), (2, 1)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    builder.append_token("second", TranslationSources.SMT, 0.1)
    builder.append_token("sentence", TranslationSources.SMT, 0.1)
    builder.mark_phrase(
        Range.create(5, 7),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    results.append(builder.to_result())

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(n=2, prefix_count=0, is_last_word_complete=True, results=results)
    assert len(suggestions) == 1
    assert list(suggestions[0].target_words) == ["this", "is", "a", "test"]


def test_get_suggestions_starts_with_punctuation() -> None:
    results: List[TranslationResult] = []
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token(",", TranslationSources.SMT, 0.5)
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=4, set_values=[(0, 1), (1, 2), (2, 3)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    results.append(builder.to_result())

    builder.reset()
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)]),
    )
    results.append(builder.to_result())

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(n=2, prefix_count=0, is_last_word_complete=True, results=results)
    assert len(suggestions) == 1
    assert list(suggestions[0].target_words) == ["this", "is", "a", "test"]


def test_get_suggestions_below_threshold() -> None:
    builder = TranslationResultBuilder(["esto", "es", "una", "prueba", "."])
    builder.append_token("this", TranslationSources.SMT, 0.5)
    builder.append_token("is", TranslationSources.SMT, 0.5)
    builder.append_token("a", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(0, 3),
        WordAlignmentMatrix.from_word_pairs(row_count=3, column_count=3, set_values=[(0, 0), (1, 1), (2, 2)]),
    )
    builder.append_token("bad", TranslationSources.SMT, 0.1)
    builder.append_token("test", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(3, 4),
        WordAlignmentMatrix.from_word_pairs(row_count=1, column_count=2, set_values=[(0, 1)]),
    )
    builder.append_token(".", TranslationSources.SMT, 0.5)
    builder.mark_phrase(
        Range.create(4, 5),
        WordAlignmentMatrix.from_word_pairs(row_count=1, column_count=1, set_values=[(0, 0)]),
    )

    suggester = PhraseTranslationSuggester(confidence_threshold=0.2)
    suggestions = suggester.get_suggestions(
        n=1, prefix_count=0, is_last_word_complete=True, results=[builder.to_result()]
    )
    assert list(suggestions[0].target_words) == ["this", "is", "a"]

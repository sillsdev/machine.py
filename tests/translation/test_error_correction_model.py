from machine.annotations import Range
from machine.translation import ErrorCorrectionModel, TranslationResultBuilder, TranslationSources, WordAlignmentMatrix

ECM = ErrorCorrectionModel()


def test_correct_prefix_empty_uncorrected_prefix_appends_prefix() -> None:
    builder = _create_result_builder("")

    prefix = "this is a test".split()
    assert (
        ECM.correct_prefix(
            builder, uncorrected_prefix_len=len(builder.target_tokens), prefix=prefix, is_last_word_complete=True
        )
        == 4
    )
    assert len(builder.confidences) == len(prefix)
    assert builder.target_tokens == prefix
    assert len(builder.phrases) == 0


def test_correct_prefix_new_end_word_inserts_word_at_end() -> None:
    builder = _create_result_builder("this is a", 2, 3)

    prefix = "this is a test".split()
    assert (
        ECM.correct_prefix(
            builder, uncorrected_prefix_len=len(builder.target_tokens), prefix=prefix, is_last_word_complete=True
        )
        == 1
    )
    assert len(builder.confidences) == len(prefix)
    assert builder.target_tokens == prefix
    assert len(builder.phrases) == 2
    assert builder.phrases[0].target_cut == 2
    assert builder.phrases[0].alignment.column_count == 2
    assert builder.phrases[1].target_cut == 3
    assert builder.phrases[1].alignment.column_count == 1


def test_correct_prefix_substring_uncorrected_prefix_new_end_word_inserts_word_at_end() -> None:
    builder = _create_result_builder("this is a and only a test", 2, 3, 5, 7)

    prefix = "this is a test".split()
    assert ECM.correct_prefix(builder, uncorrected_prefix_len=3, prefix=prefix, is_last_word_complete=True) == 0
    assert len(builder.confidences) == 8
    assert builder.target_tokens == "this is a test and only a test".split()
    assert len(builder.phrases) == 4
    assert builder.phrases[0].target_cut == 2
    assert builder.phrases[0].alignment.column_count == 2
    assert builder.phrases[1].target_cut == 3
    assert builder.phrases[1].alignment.column_count == 1
    assert builder.phrases[2].target_cut == 6
    assert builder.phrases[2].alignment.column_count == 3
    assert builder.phrases[3].target_cut == 8
    assert builder.phrases[3].alignment.column_count == 2


def test_correct_prefix_new_middle_word_inserts_word() -> None:
    builder = _create_result_builder("this is a test", 2, 4)

    prefix = "this is , a test".split()
    assert (
        ECM.correct_prefix(
            builder, uncorrected_prefix_len=len(builder.target_tokens), prefix=prefix, is_last_word_complete=True
        )
        == 0
    )
    assert len(builder.confidences) == len(prefix)
    assert builder.target_tokens == prefix
    assert len(builder.phrases) == 2
    assert builder.phrases[0].target_cut == 2
    assert builder.phrases[0].alignment.column_count == 2
    assert builder.phrases[1].target_cut == 5
    assert builder.phrases[1].alignment.column_count == 3


def test_correct_prefix_new_start_word_inserts_word_at_beginning() -> None:
    builder = _create_result_builder("this is a test", 2, 4)

    prefix = "yes this is a test".split()
    assert (
        ECM.correct_prefix(
            builder, uncorrected_prefix_len=len(builder.target_tokens), prefix=prefix, is_last_word_complete=True
        )
        == 0
    )
    assert len(builder.confidences) == len(prefix)
    assert builder.target_tokens == prefix
    assert len(builder.phrases) == 2
    assert builder.phrases[0].target_cut == 3
    assert builder.phrases[0].alignment.column_count == 3
    assert builder.phrases[1].target_cut == 5
    assert builder.phrases[1].alignment.column_count == 2


def test_correct_prefix_missing_end_word_deletes_world_at_end() -> None:
    builder = _create_result_builder("this is a test", 2, 4)

    prefix = "this is a".split()
    assert (
        ECM.correct_prefix(
            builder, uncorrected_prefix_len=len(builder.target_tokens), prefix=prefix, is_last_word_complete=True
        )
        == 0
    )
    assert len(builder.confidences) == len(prefix)
    assert builder.target_tokens == prefix
    assert len(builder.phrases) == 2
    assert builder.phrases[0].target_cut == 2
    assert builder.phrases[0].alignment.column_count == 2
    assert builder.phrases[1].target_cut == 3
    assert builder.phrases[1].alignment.column_count == 1


def test_correct_prefix_substring_uncorrected_prefix_missing_end_word_deletes_word_at_end() -> None:
    builder = _create_result_builder("this is a test and only a test", 2, 4, 6, 8)

    prefix = "this is a".split()
    assert ECM.correct_prefix(builder, uncorrected_prefix_len=4, prefix=prefix, is_last_word_complete=True) == 0
    assert len(builder.confidences) == 7
    assert builder.target_tokens == "this is a and only a test".split()
    assert len(builder.phrases) == 4
    assert builder.phrases[0].target_cut == 2
    assert builder.phrases[0].alignment.column_count == 2
    assert builder.phrases[1].target_cut == 3
    assert builder.phrases[1].alignment.column_count == 1
    assert builder.phrases[2].target_cut == 5
    assert builder.phrases[2].alignment.column_count == 2
    assert builder.phrases[3].target_cut == 7
    assert builder.phrases[3].alignment.column_count == 2


def test_correct_prefix_missing_middle_word_deletes_word() -> None:
    builder = _create_result_builder("this is a test", 2, 4)

    prefix = "this a test".split()
    assert (
        ECM.correct_prefix(
            builder, uncorrected_prefix_len=len(builder.target_tokens), prefix=prefix, is_last_word_complete=True
        )
        == 0
    )
    assert len(builder.confidences) == len(prefix)
    assert builder.target_tokens == prefix
    assert len(builder.phrases) == 2
    assert builder.phrases[0].target_cut == 1
    assert builder.phrases[0].alignment.column_count == 1
    assert builder.phrases[1].target_cut == 3
    assert builder.phrases[1].alignment.column_count == 2


def test_correct_prefix_missing_start_word_deletes_word_at_beginning() -> None:
    builder = _create_result_builder("yes this is a test", 3, 5)

    prefix = "this is a test".split()
    assert (
        ECM.correct_prefix(
            builder, uncorrected_prefix_len=len(builder.target_tokens), prefix=prefix, is_last_word_complete=True
        )
        == 0
    )
    assert len(builder.confidences) == len(prefix)
    assert builder.target_tokens == prefix
    assert len(builder.phrases) == 2
    assert builder.phrases[0].target_cut == 2
    assert builder.phrases[0].alignment.column_count == 2
    assert builder.phrases[1].target_cut == 4
    assert builder.phrases[1].alignment.column_count == 2


def _create_result_builder(target: str, *cuts: int) -> TranslationResultBuilder:
    builder = TranslationResultBuilder("esto es una prueba".split())
    if target != "":
        i = 0
        k = 0
        words = target.split()
        for j in range(len(words)):
            builder.append_token(words[j], TranslationSources.SMT, 1)
            cut = j + 1
            if k < len(cuts) and cuts[k] == cut:
                length = cut - i
                builder.mark_phrase(Range.create(i, cut), WordAlignmentMatrix.from_word_pairs(length, length))
                k += 1
                i = cut
    return builder

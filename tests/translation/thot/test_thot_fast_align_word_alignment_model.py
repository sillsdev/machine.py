from pytest import approx
from testutils.thot_test_helpers import TOY_CORPUS_FAST_ALIGN_PATH

from machine.translation import WordAlignmentMatrix
from machine.translation.thot import ThotFastAlignWordAlignmentModel, ThotSymmetrizedWordAlignmentModel

DIRECT_MODEL_PATH = TOY_CORPUS_FAST_ALIGN_PATH / "tm" / "src_trg_invswm"
INVERSE_MODEL_PATH = TOY_CORPUS_FAST_ALIGN_PATH / "tm" / "src_trg_swm"


def test_get_best_alignment() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
    target_segment = "could we see another room , please ?".split()
    matrix = model.get_best_alignment(source_segment, target_segment)
    assert matrix == WordAlignmentMatrix.from_word_pairs(9, 8, {(0, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 6), (8, 7)})


def test_get_best_alignment_batch() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    segments = [
        ("voy a marcharme hoy por la tarde .".split(), "i am leaving today in the afternoon .".split()),
        ("hablé hasta cinco en punto .".split(), "i am staying until five o ' clock .".split()),
    ]
    matrices = [matrix for _, _, matrix in model.get_best_alignment_batch(segments)]
    assert matrices == [
        WordAlignmentMatrix.from_word_pairs(8, 8, {(0, 0), (0, 1), (2, 2), (3, 3), (6, 4), (5, 5), (6, 6), (7, 7)}),
        WordAlignmentMatrix.from_word_pairs(6, 9, {(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6), (4, 7), (5, 8)}),
    ]


def test_get_avg_translation_score() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
    target_segment = "could we see another room , please ?".split()
    matrix = model.get_best_alignment(source_segment, target_segment)
    score = model.get_avg_translation_score(source_segment, target_segment, matrix)
    assert score == approx(0.34, abs=0.01)


def test_get_translation_probability() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    assert model.get_translation_probability("esto", "this") == approx(0.0, abs=0.01)
    assert model.get_translation_probability("es", "is") == approx(0.90, abs=0.01)
    assert model.get_translation_probability("una", "a") == approx(0.83, abs=0.01)
    assert model.get_translation_probability("prueba", "test") == approx(0.0, abs=0.01)


def test_source_words_enumerate() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    assert sum(1 for _ in model.source_words) == 500


def test_source_words_index_accessor() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    assert model.source_words[0] == "NULL"
    assert model.source_words[499] == "pagar"


def test_source_words_index_len() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    assert len(model.source_words) == 500


def test_target_words_enumerate() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    assert sum(1 for _ in model.target_words) == 352


def test_target_words_index_accessor() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    assert model.target_words[0] == "NULL"
    assert model.target_words[351] == "pay"


def test_target_words_index_len() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    assert len(model.target_words) == 352


def test_get_translation_table_symmetrized_no_threshold() -> None:
    model = ThotSymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH), ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH)
    )

    table = model.get_translation_table()
    assert len(table) == 500
    assert len(table["es"]) == 21


def test_get_translation_table_symmetrized_threshold() -> None:
    model = ThotSymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH), ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH)
    )

    table = model.get_translation_table(0.2)
    assert len(table) == 500
    assert len(table["es"]) == 2


def test_get_avg_translation_score_symmetrized() -> None:
    model = ThotSymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH), ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH)
    )
    source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
    target_segment = "could we see another room , please ?".split()
    matrix = model.get_best_alignment(source_segment, target_segment)
    score = model.get_avg_translation_score(source_segment, target_segment, matrix)
    assert score == approx(0.36, abs=0.01)

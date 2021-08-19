from pytest import approx
from thot_test_helpers import TOY_CORPUS_FAST_ALIGN_PATH

from machine.translation import SymmetrizedWordAlignmentModel, WordAlignmentMatrix
from machine.translation.thot import ThotFastAlignWordAlignmentModel

DIRECT_MODEL_PATH = TOY_CORPUS_FAST_ALIGN_PATH / "tm" / "src_trg_invswm"
INVERSE_MODEL_PATH = TOY_CORPUS_FAST_ALIGN_PATH / "tm" / "src_trg_swm"


def test_get_best_alignment() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
    target_segment = "could we see another room , please ?".split()
    matrix = model.get_best_alignment(source_segment, target_segment)
    assert matrix == WordAlignmentMatrix(9, 8, {(0, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 6), (8, 7)})


def test_get_best_alignments() -> None:
    model = ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH)
    source_segments = ["voy a marcharme hoy por la tarde .".split(), "hablé hasta cinco en punto .".split()]
    target_segments = ["i am leaving today in the afternoon .".split(), "i am staying until five o ' clock .".split()]
    matrices = model.get_best_alignments(source_segments, target_segments)
    assert matrices == [
        WordAlignmentMatrix(8, 8, {(0, 0), (0, 1), (2, 2), (3, 3), (6, 4), (5, 5), (6, 6), (7, 7)}),
        WordAlignmentMatrix(6, 9, {(3, 1), (1, 3), (2, 4), (4, 5), (4, 6), (4, 7), (5, 8)}),
    ]


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
    model = SymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH), ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH)
    )

    table = model.get_translation_table()
    assert len(table) == 500
    assert len(table["es"]) == 21


def test_get_translation_table_symmetrized_threshold() -> None:
    model = SymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH), ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH)
    )

    table = model.get_translation_table(0.2)
    assert len(table) == 500
    assert len(table["es"]) == 2

from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import approx, raises
from testutils.thot_test_helpers import TOY_CORPUS_FAST_ALIGN_PATH

from machine.translation import WordAlignmentMatrix
from machine.translation.thot import ThotFastAlignWordAlignmentModel, ThotSymmetrizedWordAlignmentModel

DIRECT_MODEL_PATH = TOY_CORPUS_FAST_ALIGN_PATH / "tm" / "src_trg_invswm"
INVERSE_MODEL_PATH = TOY_CORPUS_FAST_ALIGN_PATH / "tm" / "src_trg_swm"


def test_get_best_alignment() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
        target_segment = "could we see another room , please ?".split()
        matrix = model.align(source_segment, target_segment)
        assert matrix == WordAlignmentMatrix.from_word_pairs(
            9, 8, {(0, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 6), (8, 7)}
        )


def test_get_best_alignment_batch() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        segments = [
            ("voy a marcharme hoy por la tarde .".split(), "i am leaving today in the afternoon .".split()),
            ("hablé hasta cinco en punto .".split(), "i am staying until five o ' clock .".split()),
        ]
        matrices = model.align_batch(segments)
        assert matrices == [
            WordAlignmentMatrix.from_word_pairs(8, 8, {(0, 0), (0, 1), (2, 2), (3, 3), (6, 4), (5, 5), (6, 6), (7, 7)}),
            WordAlignmentMatrix.from_word_pairs(6, 9, {(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6), (4, 7), (5, 8)}),
        ]


def test_get_avg_translation_score() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
        target_segment = "could we see another room , please ?".split()
        matrix = model.align(source_segment, target_segment)
        score = model.get_avg_translation_score(source_segment, target_segment, matrix)
        assert score == approx(0.34, abs=0.01)


def test_get_translation_probability() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert model.get_translation_probability("esto", "this") == approx(0.0, abs=0.01)
        assert model.get_translation_probability("es", "is") == approx(0.90, abs=0.01)
        assert model.get_translation_probability("una", "a") == approx(0.83, abs=0.01)
        assert model.get_translation_probability("prueba", "test") == approx(0.0, abs=0.01)


def test_source_words_enumerate() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert sum(1 for _ in model.source_words) == 500


def test_source_words_index_accessor() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert model.source_words[0] == "NULL"
        assert model.source_words[499] == "pagar"


def test_source_words_index_len() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert len(model.source_words) == 500


def test_target_words_enumerate() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert sum(1 for _ in model.target_words) == 352


def test_target_words_index_accessor() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert model.target_words[0] == "NULL"
        assert model.target_words[351] == "pay"


def test_target_words_index_len() -> None:
    with ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert len(model.target_words) == 352


def test_get_translation_table_symmetrized_no_threshold() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH),
        ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH),
    ) as model:
        table = model.get_translation_table()
        assert len(table) == 500
        assert len(table["es"]) == 21


def test_get_translation_table_symmetrized_threshold() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH),
        ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH),
    ) as model:
        table = model.get_translation_table(0.2)
        assert len(table) == 500
        assert len(table["es"]) == 2


def test_get_avg_translation_score_symmetrized() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotFastAlignWordAlignmentModel(DIRECT_MODEL_PATH),
        ThotFastAlignWordAlignmentModel(INVERSE_MODEL_PATH),
    ) as model:
        source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
        target_segment = "could we see another room , please ?".split()
        matrix = model.align(source_segment, target_segment)
        score = model.get_avg_translation_score(source_segment, target_segment, matrix)
        assert score == approx(0.36, abs=0.01)


def test_constructor_model_corrupted() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        (temp_dir_path / "src_trg_invswm.src").write_text("corrupted", encoding="utf-8")
        with raises(RuntimeError):
            ThotFastAlignWordAlignmentModel(temp_dir_path / "src_trg_invswm")

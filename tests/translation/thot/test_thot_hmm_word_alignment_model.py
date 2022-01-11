from pytest import approx
from translation.thot.thot_test_helpers import TOY_CORPUS_HMM_PATH, create_test_parallel_corpus

from machine.translation import WordAlignmentMatrix
from machine.translation.thot import ThotHmmWordAlignmentModel, ThotSymmetrizedWordAlignmentModel

DIRECT_MODEL_PATH = TOY_CORPUS_HMM_PATH / "tm" / "src_trg_invswm"
INVERSE_MODEL_PATH = TOY_CORPUS_HMM_PATH / "tm" / "src_trg_swm"


def test_get_best_alignment() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
    target_segment = "could we see another room , please ?".split()
    matrix = model.get_best_alignment(source_segment, target_segment)
    assert matrix == WordAlignmentMatrix.from_word_pairs(
        9, 8, {(0, 5), (1, 6), (3, 7), (4, 0), (4, 1), (5, 2), (6, 3), (7, 4)}
    )


def test_get_best_alignments() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    source_segments = ["voy a marcharme hoy por la tarde .".split(), "hablé hasta cinco en punto .".split()]
    target_segments = ["i am leaving today in the afternoon .".split(), "i am staying until five o ' clock .".split()]
    matrices = model.get_best_alignments(source_segments, target_segments)
    assert matrices == [
        WordAlignmentMatrix.from_word_pairs(8, 8, {(0, 0), (0, 1), (0, 2), (3, 3), (6, 4), (5, 5), (6, 6), (7, 7)}),
        WordAlignmentMatrix.from_word_pairs(6, 9, {(4, 1), (5, 2), (1, 3), (2, 4), (4, 5), (4, 6), (4, 7), (5, 8)}),
    ]


def test_get_translation_probability() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    assert model.get_translation_probability("esto", "this") == approx(0.0, abs=0.01)
    assert model.get_translation_probability("es", "is") == approx(0.65, abs=0.01)
    assert model.get_translation_probability("una", "a") == approx(0.70, abs=0.01)
    assert model.get_translation_probability("prueba", "test") == approx(0.0, abs=0.01)


def test_source_words_enumerate() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    assert sum(1 for _ in model.source_words) == 513


def test_source_words_index_accessor() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    assert model.source_words[0] == "NULL"
    assert model.source_words[512] == "pagar"


def test_source_words_index_len() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    assert len(model.source_words) == 513


def test_target_words_enumerate() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    assert sum(1 for _ in model.target_words) == 363


def test_target_words_index_accessor() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    assert model.target_words[0] == "NULL"
    assert model.target_words[362] == "pay"


def test_target_words_index_len() -> None:
    model = ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH)
    assert len(model.target_words) == 363


def test_get_translation_table_symmetrized_no_threshold() -> None:
    model = ThotSymmetrizedWordAlignmentModel(
        ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH), ThotHmmWordAlignmentModel(INVERSE_MODEL_PATH)
    )

    table = model.get_translation_table()
    assert len(table) == 513
    assert len(table["es"]) == 23


def test_get_translation_table_symmetrized_threshold() -> None:
    model = ThotSymmetrizedWordAlignmentModel(
        ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH), ThotHmmWordAlignmentModel(INVERSE_MODEL_PATH)
    )

    table = model.get_translation_table(0.2)
    assert len(table) == 513
    assert len(table["es"]) == 9


def test_create_trainer() -> None:
    model = ThotHmmWordAlignmentModel()
    model.parameters.ibm1_iteration_count = 2
    model.parameters.hmm_iteration_count = 2
    model.parameters.hmm_p0 = 0.1
    trainer = model.create_trainer(create_test_parallel_corpus())
    trainer.train()
    trainer.save()

    matrix = model.get_best_alignment("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())
    assert matrix == WordAlignmentMatrix.from_word_pairs(5, 6, {(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5)})

    matrix = model.get_best_alignment("isthay isyay otnay ayay esttay-N .".split(), "this is not a test N .".split())
    assert matrix == WordAlignmentMatrix.from_word_pairs(6, 7, {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 6)})

    matrix = model.get_best_alignment("isthay isyay ayay esttay-N ardhay .".split(), "this is a hard test N .".split())
    assert matrix == WordAlignmentMatrix.from_word_pairs(6, 7, {(0, 0), (1, 1), (2, 2), (4, 3), (3, 4), (3, 5), (3, 6)})

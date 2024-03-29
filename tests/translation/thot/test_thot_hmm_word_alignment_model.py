from pytest import approx
from testutils.thot_test_helpers import TOY_CORPUS_HMM_PATH, create_test_parallel_corpus

from machine.translation import WordAlignmentMatrix
from machine.translation.thot import ThotHmmWordAlignmentModel, ThotSymmetrizedWordAlignmentModel

DIRECT_MODEL_PATH = TOY_CORPUS_HMM_PATH / "tm" / "src_trg_invswm"
INVERSE_MODEL_PATH = TOY_CORPUS_HMM_PATH / "tm" / "src_trg_swm"


def test_get_best_alignment() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
        target_segment = "could we see another room , please ?".split()
        matrix = model.align(source_segment, target_segment)
        assert matrix == WordAlignmentMatrix.from_word_pairs(
            9, 8, {(0, 5), (1, 6), (3, 7), (4, 0), (4, 1), (5, 2), (6, 3), (7, 4)}
        )


def test_get_best_alignment_batch() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        segments = [
            ("voy a marcharme hoy por la tarde .".split(), "i am leaving today in the afternoon .".split()),
            ("hablé hasta cinco en punto .".split(), "i am staying until five o ' clock .".split()),
        ]
        matrices = model.align_batch(segments)
        assert matrices == [
            WordAlignmentMatrix.from_word_pairs(8, 8, {(0, 0), (0, 1), (0, 2), (3, 3), (6, 4), (5, 5), (6, 6), (7, 7)}),
            WordAlignmentMatrix.from_word_pairs(6, 9, {(4, 1), (5, 2), (1, 3), (2, 4), (4, 5), (4, 6), (4, 7), (5, 8)}),
        ]


def test_get_avg_translation_score() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
        target_segment = "could we see another room , please ?".split()
        matrix = model.align(source_segment, target_segment)
        score = model.get_avg_translation_score(source_segment, target_segment, matrix)
        assert score == approx(0.40, abs=0.01)


def test_get_best_aligned_word_pairs() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        source_segment = "hablé hasta cinco en punto .".split()
        target_segment = "i am staying until five o ' clock .".split()
        pairs = list(model.get_best_aligned_word_pairs(source_segment, target_segment))
        assert len(pairs) == 8

        assert pairs[0].source_index == 1
        assert pairs[0].target_index == 3
        assert pairs[0].translation_score == approx(0.78, abs=0.01)
        assert pairs[0].alignment_score == approx(0.18, abs=0.01)


def test_compute_aligned_word_pair_scores() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        source_segment = "hablé hasta cinco en punto .".split()
        target_segment = "i am staying until five o ' clock .".split()
        matrix = model.align(source_segment, target_segment)
        pairs = list(matrix.to_aligned_word_pairs(include_null=True))
        model.compute_aligned_word_pair_scores(source_segment, target_segment, pairs)
        assert len(pairs) == 11

        assert pairs[0].source_index == -1
        assert pairs[0].target_index == 0
        assert pairs[0].translation_score == approx(0.34, abs=0.01)
        assert pairs[0].alignment_score == approx(0.08, abs=0.01)

        assert pairs[1].source_index == 0
        assert pairs[1].target_index == -1
        assert pairs[1].translation_score == 0
        assert pairs[1].alignment_score == 0

        assert pairs[2].source_index == 1
        assert pairs[2].target_index == 3
        assert pairs[2].translation_score == approx(0.78, abs=0.01)
        assert pairs[2].alignment_score == approx(0.18, abs=0.01)


def test_get_translation_probability() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert model.get_translation_probability("esto", "this") == approx(0.0, abs=0.01)
        assert model.get_translation_probability("es", "is") == approx(0.65, abs=0.01)
        assert model.get_translation_probability("una", "a") == approx(0.70, abs=0.01)
        assert model.get_translation_probability("prueba", "test") == approx(0.0, abs=0.01)


def test_source_words_enumerate() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert sum(1 for _ in model.source_words) == 513


def test_source_words_index_accessor() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert model.source_words[0] == "NULL"
        assert model.source_words[512] == "pagar"


def test_source_words_index_len() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert len(model.source_words) == 513


def test_target_words_enumerate() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert sum(1 for _ in model.target_words) == 363


def test_target_words_index_accessor() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert model.target_words[0] == "NULL"
        assert model.target_words[362] == "pay"


def test_target_words_index_len() -> None:
    with ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH) as model:
        assert len(model.target_words) == 363


def test_get_translation_table_symmetrized_no_threshold() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH), ThotHmmWordAlignmentModel(INVERSE_MODEL_PATH)
    ) as model:
        table = model.get_translation_table()
        assert len(table) == 513
        assert len(table["es"]) == 23


def test_get_translation_table_symmetrized_threshold() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH), ThotHmmWordAlignmentModel(INVERSE_MODEL_PATH)
    ) as model:
        table = model.get_translation_table(0.2)
        assert len(table) == 513
        assert len(table["es"]) == 9


def test_get_avg_translation_score_symmetrized() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH), ThotHmmWordAlignmentModel(INVERSE_MODEL_PATH)
    ) as model:
        source_segment = "por favor , ¿ podríamos ver otra habitación ?".split()
        target_segment = "could we see another room , please ?".split()
        matrix = model.align(source_segment, target_segment)
        score = model.get_avg_translation_score(source_segment, target_segment, matrix)
        assert score == approx(0.46, abs=0.01)


def test_get_best_aligned_word_pairs_symmetrized() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH), ThotHmmWordAlignmentModel(INVERSE_MODEL_PATH)
    ) as model:
        source_segment = "hablé hasta cinco en punto .".split()
        target_segment = "i am staying until five o ' clock .".split()
        pairs = list(model.get_best_aligned_word_pairs(source_segment, target_segment))
        assert len(pairs) == 8

        assert pairs[0].source_index == 0
        assert pairs[0].target_index == 1
        assert pairs[0].translation_score == approx(0.01, abs=0.01)
        assert pairs[0].alignment_score == approx(0.26, abs=0.01)


def test_compute_aligned_word_pair_scores_symmetrized() -> None:
    with ThotSymmetrizedWordAlignmentModel(
        ThotHmmWordAlignmentModel(DIRECT_MODEL_PATH), ThotHmmWordAlignmentModel(INVERSE_MODEL_PATH)
    ) as model:
        source_segment = "hablé hasta cinco en punto .".split()
        target_segment = "i am staying until five o ' clock .".split()
        matrix = model.align(source_segment, target_segment)
        pairs = list(matrix.to_aligned_word_pairs(include_null=True))
        model.compute_aligned_word_pair_scores(source_segment, target_segment, pairs)
        assert len(pairs) == 10

        assert pairs[0].source_index == -1
        assert pairs[0].target_index == 0
        assert pairs[0].translation_score == approx(0.34, abs=0.01)
        assert pairs[0].alignment_score == approx(0.08, abs=0.01)

        assert pairs[1].source_index == -1
        assert pairs[1].target_index == 2
        assert pairs[1].translation_score == approx(0.01, abs=0.01)
        assert pairs[1].alignment_score == approx(0.11, abs=0.01)

        assert pairs[2].source_index == 0
        assert pairs[2].target_index == 1
        assert pairs[2].translation_score == approx(0.01, abs=0.01)
        assert pairs[2].alignment_score == approx(0.26, abs=0.01)


def test_create_trainer() -> None:
    with ThotHmmWordAlignmentModel() as model:
        model.parameters.ibm1_iteration_count = 2
        model.parameters.hmm_iteration_count = 2
        model.parameters.hmm_p0 = 0.1
        with model.create_trainer(create_test_parallel_corpus()) as trainer:
            trainer.train()
            trainer.save()

        matrix = model.align("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())
        assert matrix == WordAlignmentMatrix.from_word_pairs(5, 6, {(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5)})

        matrix = model.align("isthay isyay otnay ayay esttay-N .".split(), "this is not a test N .".split())
        assert matrix == WordAlignmentMatrix.from_word_pairs(
            6, 7, {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 6)}
        )

        matrix = model.align("isthay isyay ayay esttay-N ardhay .".split(), "this is a hard test N .".split())
        assert matrix == WordAlignmentMatrix.from_word_pairs(
            6, 7, {(0, 0), (1, 1), (2, 2), (4, 3), (3, 4), (3, 5), (3, 6)}
        )

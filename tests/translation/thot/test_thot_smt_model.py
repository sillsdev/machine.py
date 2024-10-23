from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import raises

from machine.translation.thot import ThotSmtModel, ThotSmtParameters, ThotWordAlignmentModelType
from tests.testutils.thot_test_helpers import TOY_CORPUS_FAST_ALIGN_CONFIG_FILENAME, TOY_CORPUS_HMM_CONFIG_FILENAME


def test_translate_target_segment_hmm() -> None:
    with _create_hmm_model() as smt_model:
        result = smt_model.translate("voy a marcharme hoy por la tarde .")
        assert result.translation == "i am leaving today in the afternoon ."


def test_translate_n_less_than_n_hmm() -> None:
    with _create_hmm_model() as smt_model:
        results = smt_model.translate_n(3, "voy a marcharme hoy por la tarde .")
        assert [r.translation for r in results] == ["i am leaving today in the afternoon ."]


def test_translate_n_hmm() -> None:
    with _create_hmm_model() as smt_model:
        results = smt_model.translate_n(2, "hablé hasta cinco en punto .")
        assert [r.translation for r in results] == [
            "hablé until five o ' clock .",
            "hablé until five o ' clock for",
        ]


def test_train_segment_hmm() -> None:
    with _create_hmm_model() as smt_model:
        result = smt_model.translate("esto es una prueba .")
        assert result.translation == "esto is a prueba ."
        smt_model.train_segment("esto es una prueba .", "this is a test .")
        result = smt_model.translate("esto es una prueba .")
        assert result.translation == "this is a test ."


def test_get_word_graph_empty_segment_hmm() -> None:
    with _create_hmm_model() as smt_model:
        word_graph = smt_model.get_word_graph([])
        assert word_graph.is_empty


def test_translate_batch_hmm() -> None:
    batch = [
        "por favor , desearía reservar una habitación hasta mañana .",
        "por favor , despiértenos mañana a las siete y cuarto .",
        "voy a marcharme hoy por la tarde .",
        "por favor , ¿ les importaría bajar nuestro equipaje a la habitación número cero trece ?",
        "¿ me podrían dar la llave de la habitación dos cuatro cuatro , por favor ?",
    ]

    with _create_hmm_model() as smt_model:
        results = smt_model.translate_batch(batch)
        assert [r.translation for r in results] == [
            "please i would like to book a room until tomorrow .",
            "please wake us up tomorrow at a quarter past seven .",
            "i am leaving today in the afternoon .",
            "please would you mind sending down our luggage to room number oh thirteenth ?",
            "could you give me the key to room number two four four , please ?",
        ]


def test_translate_fast_align() -> None:
    with _create_fast_align_model() as smt_model:
        result = smt_model.translate("voy a marcharme hoy por la tarde .")
        assert result.translation == "i am leaving today in the afternoon ."


def test_translate_n_less_than_n_fast_align() -> None:
    with _create_fast_align_model() as smt_model:
        results = smt_model.translate_n(3, "voy a marcharme hoy por la tarde .")
        assert [r.translation for r in results] == ["i am leaving today in the afternoon ."]


def test_translate_n_fast_align() -> None:
    with _create_fast_align_model() as smt_model:
        results = smt_model.translate_n(2, "hablé hasta cinco en punto .")
        assert [r.translation for r in results] == [
            "hablé until five o ' clock .",
            "hablé until five o ' clock , please .",
        ]


def test_train_segment_fast_align() -> None:
    with _create_fast_align_model() as smt_model:
        result = smt_model.translate("esto es una prueba .")
        assert result.translation == "esto is a prueba ."
        smt_model.train_segment("esto es una prueba .", "this is a test .")
        result = smt_model.translate("esto es una prueba .")
        assert result.translation == "this is a test ."


def test_get_word_graph_empty_segment_fast_align() -> None:
    with _create_fast_align_model() as smt_model:
        word_graph = smt_model.get_word_graph([])
        assert word_graph.is_empty


def test_constructor_model_not_found() -> None:
    with raises(FileNotFoundError):
        ThotSmtModel(
            ThotWordAlignmentModelType.HMM,
            ThotSmtParameters(
                translation_model_filename_prefix="does-not-exist",
                language_model_filename_prefix="does-not-exist",
            ),
        )


def test_constructor_model_corrupted() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        tm_dir_path = temp_dir_path / "tm"
        tm_dir_path.mkdir()
        (tm_dir_path / "src_trg.ttable").write_text("corrupted", encoding="utf-8")
        lm_dir_path = temp_dir_path / "lm"
        lm_dir_path.mkdir()
        (lm_dir_path / "trg.lm").write_text("corrupted", encoding="utf-8")
        with raises(RuntimeError):
            ThotSmtModel(
                ThotWordAlignmentModelType.HMM,
                ThotSmtParameters(
                    translation_model_filename_prefix=str(tm_dir_path / "src_trg"),
                    language_model_filename_prefix=str(lm_dir_path / "trg.lm"),
                ),
            )


def _create_hmm_model() -> ThotSmtModel:
    return ThotSmtModel(ThotWordAlignmentModelType.HMM, TOY_CORPUS_HMM_CONFIG_FILENAME)


def _create_fast_align_model() -> ThotSmtModel:
    return ThotSmtModel(ThotWordAlignmentModelType.FAST_ALIGN, TOY_CORPUS_FAST_ALIGN_CONFIG_FILENAME)


def test_get_word_graph_hmm() -> None:
    with _create_hmm_model() as smt_model:
        word_graph = smt_model.get_word_graph("voy a marcharme hoy por la tarde .")
        assert len(word_graph.arcs) == 2
        assert len(word_graph.final_states) == 1


def test_get_word_graph_fast_align() -> None:
    with _create_fast_align_model() as smt_model:
        word_graph = smt_model.get_word_graph("voy a marcharme hoy por la tarde .")
        assert len(word_graph.arcs) == 2
        assert len(word_graph.final_states) == 1

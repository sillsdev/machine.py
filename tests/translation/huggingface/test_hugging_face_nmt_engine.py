import sys

if sys.platform == "darwin":
    from pytest import skip

    skip("skipping Hugging Face tests on MacOS", allow_module_level=True)

from pytest import approx, mark, raises

from machine.translation.huggingface import HuggingFaceNmtEngine


@mark.parametrize("output_attentions", [True, False])
def test_translate_n_batch_beam(output_attentions: bool) -> None:
    with HuggingFaceNmtEngine(
        "stas/tiny-m2m_100",
        src_lang="en",
        tgt_lang="es",
        num_beams=2,
        max_length=10,
        output_attentions=output_attentions,
    ) as engine:
        results = engine.translate_n_batch(
            n=2,
            segments=["This is a test string", "Hello, world!"],
        )
        assert results[0][0].translation == "skaberskaber Dollar Dollar ፤ ፤ gerekir gerekir"
        assert results[0][0].confidences[0] == approx(1.08e-05, 0.01)
        assert str(results[0][0].alignment) == ("2-0 2-1 2-2 2-3 4-4 4-5 4-6 4-7" if output_attentions else "")
        assert results[0][1].translation == "skaberskaber Dollar Dollar ፤ ፤ ፤ gerekir"
        assert results[0][1].confidences[0] == approx(1.08e-05, 0.01)
        assert str(results[0][1].alignment) == ("2-0 2-1 2-2 2-3 4-4 4-5 4-6 4-7" if output_attentions else "")
        assert results[1][0].translation == "skaberskaber Dollar Dollar ፤ ፤ gerekir gerekir"
        assert results[1][0].confidences[0] == approx(1.08e-05, 0.01)
        assert str(results[1][0].alignment) == ("0-1 0-2 0-7 1-0 3-3 3-4 3-5 3-6" if output_attentions else "")
        assert results[1][1].translation == "skaberskaber Dollar Dollar ፤ ፤ ፤ gerekir"
        assert results[1][1].confidences[0] == approx(1.08e-05, 0.01)
        assert str(results[1][1].alignment) == ("0-1 0-2 0-7 1-0 3-3 3-4 3-5 3-6" if output_attentions else "")


@mark.parametrize("output_attentions", [True, False])
def test_translate_greedy(output_attentions: bool) -> None:
    with HuggingFaceNmtEngine(
        "stas/tiny-m2m_100", src_lang="en", tgt_lang="es", max_length=10, output_attentions=output_attentions
    ) as engine:
        result = engine.translate("This is a test string")
        assert result.translation == "skaberskaber Dollar Dollar Dollar ፤ gerekir gerekir"
        assert result.confidences[0] == approx(1.08e-05, 0.01)
        assert str(result.alignment) == ("2-0 2-1 2-2 2-3 4-4 4-5 4-6 4-7" if output_attentions else "")


@mark.parametrize("output_attentions", [True, False])
def test_construct_invalid_lang(output_attentions: bool) -> None:
    with raises(ValueError):
        HuggingFaceNmtEngine("stas/tiny-m2m_100", src_lang="qaa", tgt_lang="es", output_attentions=output_attentions)

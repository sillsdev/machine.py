import sys

if sys.platform == "darwin":
    from pytest import skip

    skip("skipping Hugging Face tests on MacOS", allow_module_level=True)

from math import exp, log
from typing import Any, Dict, Optional

from pytest import approx, mark, raises
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from machine.translation.huggingface import HuggingFaceNmtEngine, SilTranslationPipeline
from machine.translation.translation_result import TranslationResult


@mark.parametrize("output_attentions", [True, False])
def test_translate_n_batch_beam(output_attentions: bool) -> None:
    with HuggingFaceNmtEngine(
        "stas/tiny-m2m_100",
        src_lang="en",
        tgt_lang="es",
        num_beams=2,
        max_length=10,
        device="cpu",  # Keep test results consistent across platforms
        output_attentions=output_attentions,
    ) as engine:
        results = engine.translate_n_batch(
            n=2,
            segments=["This is a test string", "Hello, world!"],
        )
        assert results[0][0].translation == "skaberskaber Dollar Dollar ፤ ፤ gerekir gerekir"
        assert results[0][0].confidences[0] == approx(1.08e-05, 0.01)
        assert results[0][0].sequence_confidence == approx(_get_sequence_confidence(results[0][0]), 0.01)
        assert str(results[0][0].alignment) == ("0-2 2-0 2-1 2-3 4-4 4-5 4-6 4-7" if output_attentions else "")

        assert results[0][1].translation == "skaberskaber Dollar Dollar ፤ ፤ ፤ gerekir"
        assert results[0][1].confidences[0] == approx(1.08e-05, 0.01)
        assert results[0][1].sequence_confidence == approx(_get_sequence_confidence(results[0][0]), 0.01)
        assert str(results[0][1].alignment) == ("0-2 2-0 2-1 2-3 4-4 4-5 4-6 4-7" if output_attentions else "")

        assert results[1][0].translation == "skaberskaber Dollar Dollar ፤ ፤ gerekir gerekir"
        assert results[1][0].confidences[0] == approx(1.08e-05, 0.01)
        assert str(results[1][0].alignment) == ("0-0 0-1 0-2 0-3 0-7 3-4 3-5 3-6" if output_attentions else "")
        assert results[1][0].sequence_confidence == approx(_get_sequence_confidence(results[0][0]), 0.01)

        assert results[1][1].translation == "skaberskaber Dollar Dollar ፤ ፤ ፤ gerekir"
        assert results[1][1].confidences[0] == approx(1.08e-05, 0.01)
        assert str(results[1][1].alignment) == ("0-0 0-1 0-2 0-3 0-7 3-4 3-5 3-6" if output_attentions else "")
        assert results[1][1].sequence_confidence == approx(_get_sequence_confidence(results[0][0]), 0.01)


@mark.parametrize("output_attentions", [True, False])
def test_translate_greedy(output_attentions: bool) -> None:
    with HuggingFaceNmtEngine(
        "stas/tiny-m2m_100",
        src_lang="en",
        tgt_lang="es",
        max_length=10,
        device="cpu",  # Keep test results consistent across platforms
        output_attentions=output_attentions,
    ) as engine:
        result = engine.translate("This is a test string")
        assert result.translation == "skaberskaber Dollar Dollar Dollar ፤ gerekir gerekir"
        assert result.confidences[0] == approx(1.08e-05, 0.01)
        # Greedy search does not produce a sequence score
        assert result.sequence_confidence == -1.0
        assert str(result.alignment) == ("0-2 2-0 2-1 2-3 4-4 4-5 4-6 4-7" if output_attentions else "")


@mark.parametrize("output_attentions", [True, False])
def test_construct_invalid_lang(output_attentions: bool) -> None:
    with raises(ValueError):
        HuggingFaceNmtEngine("stas/tiny-m2m_100", src_lang="qaa", tgt_lang="es", output_attentions=output_attentions)


def test_construct_invalid_lang_leaves_caller_model_unchanged() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="sdpa")
    with raises(ValueError):
        HuggingFaceNmtEngine(model, src_lang="en", tgt_lang="qaa", output_attentions=True)
    assert model.config._attn_implementation == "sdpa"


def test_close_restores_caller_model_attn_implementation() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="sdpa")
    with HuggingFaceNmtEngine(model, src_lang="en", tgt_lang="es", max_length=10, output_attentions=True) as engine:
        assert model.config._attn_implementation == "eager"
        assert str(engine.translate("This is a test string").alignment) != ""
    assert model.config._attn_implementation == "sdpa"


@mark.parametrize("output_attentions", [True, False])
def test_translate_model_without_sdpa_support(output_attentions: bool) -> None:
    with HuggingFaceNmtEngine(
        "hf-internal-testing/tiny-random-T5ForConditionalGeneration",
        max_length=10,
        device="cpu",
        output_attentions=output_attentions,
    ) as engine:
        result = engine.translate("This is a test string")
        assert result.source_tokens == ["▁This", "▁is", "▁", "a", "▁test", "▁string"]


@mark.parametrize(
    "construct_output_attentions, call_output_attentions, expect_attentions",
    [(None, None, False), (True, False, False), (False, True, True)],
)
def test_pipeline_output_attentions(
    construct_output_attentions: Optional[bool], call_output_attentions: Optional[bool], expect_attentions: bool
) -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("stas/tiny-m2m_100")
    construct_kwargs: Dict[str, Any] = (
        {} if construct_output_attentions is None else {"output_attentions": construct_output_attentions}
    )
    call_kwargs = {} if call_output_attentions is None else {"output_attentions": call_output_attentions}
    pipeline = SilTranslationPipeline(
        model=model, tokenizer=tokenizer, batch_size=1, src_lang="en", tgt_lang="es", max_length=10, **construct_kwargs
    )

    outputs = pipeline(["This is a test string"], **call_kwargs)

    assert ("token_attentions" in outputs[0]) == expect_attentions


def _get_sequence_confidence(result: TranslationResult) -> float:
    # Inject a 0 score for the BOS token
    return exp(sum([log(c) for c in result.confidences] + [0]) / (len(result.confidences) + 1))

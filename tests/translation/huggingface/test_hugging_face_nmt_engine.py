import sys

if sys.platform == "darwin":
    from pytest import skip

    skip("skipping Hugging Face tests on MacOS", allow_module_level=True)

from math import exp, log
from typing import Any, Dict, Optional

import torch
from pytest import approx, mark, raises
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from machine.translation.huggingface import HuggingFaceNmtEngine, SilTranslationPipeline
from machine.translation.huggingface.hugging_face_nmt_engine import _compute_transition_scores
from machine.translation.translation_result import TranslationResult
from machine.translation.word_alignment_matrix import WordAlignmentMatrix


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


@mark.parametrize("output_attentions", [True, None])
def test_construct_output_attentions_non_eager_caller_model(output_attentions: Optional[bool]) -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="sdpa")
    with raises(ValueError, match="eager attention"):
        HuggingFaceNmtEngine(model, src_lang="en", tgt_lang="es", output_attentions=output_attentions)
    assert model.config._attn_implementation == "sdpa"


def test_translate_non_eager_caller_model_without_output_attentions() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="sdpa")
    with HuggingFaceNmtEngine(model, src_lang="en", tgt_lang="es", max_length=10, output_attentions=False) as engine:
        result = engine.translate("This is a test string")
    assert result.translation == "skaberskaber Dollar Dollar Dollar ፤ gerekir gerekir"
    assert str(result.alignment) == ""
    assert model.config._attn_implementation == "sdpa"


def test_engines_share_caller_model() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="eager")
    engine1 = HuggingFaceNmtEngine(model, src_lang="en", tgt_lang="es", max_length=10, output_attentions=True)
    with HuggingFaceNmtEngine(model, src_lang="en", tgt_lang="es", max_length=10, output_attentions=True) as engine2:
        engine1.close()
        assert str(engine2.translate("This is a test string").alignment) == "0-2 2-0 2-1 2-3 4-4 4-5 4-6 4-7"


def test_translate_greedy_alignment_matches_cross_attention() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("stas/tiny-m2m_100", src_lang="en", tgt_lang="es")
    with HuggingFaceNmtEngine(model, src_lang="en", tgt_lang="es", max_length=10, device="cpu") as engine:
        result = engine.translate("This is a test string")

    inputs = tokenizer("This is a test string", return_tensors="pt")
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id("es"),
        max_length=10,
        output_attentions=True,
        return_dict_in_generate=True,
    )
    assert tokenizer.decode(output.sequences[0], skip_special_tokens=True) == result.translation

    # The engine aligns each target token to the source token with the most cross-attention, averaged over the
    # heads of the layer two thirds of the way through the decoder.
    special_ids = set(tokenizer.all_special_ids)
    src_indices = [i for i, id in enumerate(inputs["input_ids"][0].tolist()) if id not in special_ids]
    layer = (2 * len(output.cross_attentions[0])) // 3
    word_pairs = []
    for i, id in enumerate(output.sequences[0].tolist()):
        if id in special_ids:
            continue
        # The sequence starts with the decoder start token, so token i is generated at step i - 1.
        attention = output.cross_attentions[i - 1][layer][0].mean(dim=0)[-1, src_indices]
        word_pairs.append((int(attention.argmax()), len(word_pairs)))
    expected = WordAlignmentMatrix.from_word_pairs(len(src_indices), len(word_pairs), word_pairs)
    assert str(result.alignment) == str(expected)


@mark.parametrize("num_beams", [1, 2])
def test_compute_transition_scores_matches_transformers(num_beams: int) -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100")
    tokenizer = AutoTokenizer.from_pretrained("stas/tiny-m2m_100", src_lang="en", tgt_lang="es")
    inputs = tokenizer(["This is a test string", "Hello, world!"], return_tensors="pt", padding=True)
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id("es"),
        max_length=10,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        output_scores=True,
        return_dict_in_generate=True,
    )
    beam_indices = output.beam_indices if num_beams > 1 else None
    # Beam search scores are already log probabilities, and greedy search scores are logits.
    normalize_logits = num_beams == 1

    expected = model.compute_transition_scores(
        output.sequences, output.scores, beam_indices, normalize_logits=normalize_logits
    )
    actual = _compute_transition_scores(output.sequences, output.scores, beam_indices, normalize_logits)

    assert torch.allclose(actual, expected)


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
    [(None, None, False), (True, None, True), (True, False, False), (False, True, True)],
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


def test_pipeline_output_attentions_non_eager_model() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained("stas/tiny-m2m_100", attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained("stas/tiny-m2m_100")
    with raises(ValueError, match="eager attention"):
        SilTranslationPipeline(model=model, tokenizer=tokenizer, batch_size=1, output_attentions=True)

    pipeline = SilTranslationPipeline(
        model=model, tokenizer=tokenizer, batch_size=1, src_lang="en", tgt_lang="es", max_length=10
    )
    with raises(ValueError, match="eager attention"):
        pipeline(["This is a test string"], output_attentions=True)


def _get_sequence_confidence(result: TranslationResult) -> float:
    # Inject a 0 score for the BOS token
    return exp(sum([log(c) for c in result.confidences] + [0]) / (len(result.confidences) + 1))

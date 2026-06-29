from typing import Optional

import pytest
import thot.alignment as ta
from testutils.thot_test_helpers import create_test_parallel_corpus

from machine.translation import WordAlignmentMatrix
from machine.translation.thot import (
    ThotEflomalWordAlignmentModel,
    ThotWordAlignmentParameters,
    create_thot_symmetrized_word_alignment_model,
)

# Eflomal requires a Thot build that includes EflomalAlignmentModel (sil-thot >= 3.5.0).
pytestmark = pytest.mark.skipif(
    not hasattr(ta, "EflomalAlignmentModel"),
    reason="Eflomal requires a Thot build that includes EflomalAlignmentModel",
)


def _train_model(
    prefix_filename: Optional[str] = None, parameters: Optional[ThotWordAlignmentParameters] = None
) -> ThotEflomalWordAlignmentModel:
    # When no iteration counts are specified, Eflomal derives its schedule automatically from
    # the corpus size (the recommended default).
    model = ThotEflomalWordAlignmentModel(prefix_filename)
    if parameters is not None:
        model.parameters = parameters
    with model.create_trainer(create_test_parallel_corpus()) as trainer:
        trainer.train()
        trainer.save()
    return model


def test_create_trainer_and_align() -> None:
    with _train_model() as model:
        matrix = model.align("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())
        # The Bayesian sampler is deterministic for a fixed seed, but we assert structural
        # plausibility rather than an exact matrix so the test is robust to schedule tuning.
        assert matrix.row_count == 5
        assert matrix.column_count == 6
        assert matrix[0, 0]  # "this" aligns to "isthay"
        assert len(matrix.to_aligned_word_pairs()) > 0


def test_align_batch() -> None:
    with _train_model() as model:
        segments = [
            ("isthay isyay ayay esttay-N .".split(), "this is a test N .".split()),
            ("ouyay ouldshay esttay-V oftenyay .".split(), "you should test V often .".split()),
            ("isyay isthay orkingway ?".split(), "is this working ?".split()),
        ]
        matrices = model.align_batch(segments)
        assert len(matrices) == 3
        assert all(len(m.to_aligned_word_pairs()) > 0 for m in matrices)


def test_get_translation_probability() -> None:
    with _train_model() as model:
        # After training the model should strongly associate the obvious translation.
        assert model.get_translation_probability("isthay", "this") > 0.1


def test_deterministic_training_is_reproducible() -> None:
    def train_deterministic() -> WordAlignmentMatrix:
        parameters = ThotWordAlignmentParameters(eflomal_deterministic=True)
        with _train_model(parameters=parameters) as model:
            return model.align("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())

    # With deterministic training the chains run serially from a fixed seed, so two separate
    # training runs produce an identical model.
    first = train_deterministic()
    second = train_deterministic()
    assert first == second


def test_create_trainer_with_explicit_schedule() -> None:
    # Specifying the IBM1/HMM/IBM3 iteration counts overrides the automatic schedule with an
    # explicit IBM1 -> HMM -> fertility (IBM3) schedule.
    parameters = ThotWordAlignmentParameters(ibm1_iteration_count=50, hmm_iteration_count=50, ibm3_iteration_count=200)
    with _train_model(parameters=parameters) as model:
        matrix = model.align("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())
        assert matrix[0, 0]  # "this" aligns to "isthay"
        assert model.get_translation_probability("isthay", "this") > 0.1


def test_source_and_target_words() -> None:
    with _train_model() as model:
        assert len(model.source_words) > 0
        assert len(model.target_words) > 0


def test_save_and_load_round_trip(tmp_path) -> None:
    prefix = str(tmp_path / "src_trg_invswm")

    with _train_model(prefix) as model:
        trained = model.align("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())

    with ThotEflomalWordAlignmentModel(prefix) as loaded:
        reloaded = loaded.align("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())

    assert reloaded == trained


def test_symmetrized_alignment() -> None:
    model = create_thot_symmetrized_word_alignment_model("eflomal")
    with model:
        with model.create_trainer(create_test_parallel_corpus()) as trainer:
            trainer.train()
            trainer.save()

        matrix = model.align("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())
        assert len(matrix.to_aligned_word_pairs()) > 0

from typing import Iterable, Optional

import pytest
from decoy import Decoy, matchers
from testutils.thot_test_helpers import create_test_parallel_corpus

from machine.corpora import (
    AlignedWordPair,
    DictionaryTextCorpus,
    MemoryText,
    ParallelTextCorpus,
    StandardParallelTextCorpus,
    TextRow,
)
from machine.translation import SymmetrizationHeuristic, Trainer, WordAlignmentMatrix, thot, word_align_corpus
from machine.translation.thot import ThotSymmetrizedWordAlignmentModel, create_thot_symmetrized_word_alignment_model

_ANY = matchers.AnythingOrNone()


def _alignment_strings(corpus: ParallelTextCorpus, text_ids: Optional[Iterable[str]] = None) -> list:
    return [
        AlignedWordPair.to_string(row.aligned_word_pairs, include_scores=False) for row in corpus.get_rows(text_ids)
    ]


def _create_trained_model(corpus: ParallelTextCorpus, aligner: str = "fast_align") -> ThotSymmetrizedWordAlignmentModel:
    model = create_thot_symmetrized_word_alignment_model(aligner)
    model.heuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND  # word_align_corpus's default
    model.emit_training_alignments = True
    with model.create_trainer(corpus) as trainer:
        trainer.train()
        trainer.save()
    return model


@pytest.mark.parametrize("aligner", ["fast_align", "ibm1"])
def test_word_align_corpus_transductive_matches_inference(aligner: str) -> None:
    # For deterministic models, the alignments retained during training match those produced by a
    # separate inference pass, so the transductive output must equal aligning each row directly.
    transductive = _alignment_strings(word_align_corpus(create_test_parallel_corpus(), aligner=aligner))

    with _create_trained_model(create_test_parallel_corpus(), aligner) as model:
        inference = [
            AlignedWordPair.to_string(
                model.align(row.source_segment, row.target_segment).to_aligned_word_pairs(), include_scores=False
            )
            for row in create_test_parallel_corpus().get_rows()
        ]
    assert transductive == inference


def test_word_align_corpus_default_is_transductive() -> None:
    rows = list(word_align_corpus(create_test_parallel_corpus()).get_rows())
    assert len(rows) == 8
    assert any(row.aligned_word_pairs for row in rows)


def _create_two_text_parallel_corpus() -> StandardParallelTextCorpus:
    src = DictionaryTextCorpus(
        MemoryText("text1", [TextRow("text1", 1, "el gato".split()), TextRow("text1", 2, "la casa".split())]),
        MemoryText("text2", [TextRow("text2", 1, "el perro corre".split()), TextRow("text2", 2, "la mesa".split())]),
    )
    trg = DictionaryTextCorpus(
        MemoryText("text1", [TextRow("text1", 1, "the cat".split()), TextRow("text1", 2, "the house".split())]),
        MemoryText("text2", [TextRow("text2", 1, "the dog runs".split()), TextRow("text2", 2, "the table".split())]),
    )
    return StandardParallelTextCorpus(src, trg)


def test_word_align_corpus_transductive_text_ids_keep_index_in_sync() -> None:
    # Filtering by text must not desync the training-alignment index: the rows for a requested text
    # must get exactly the alignments they got in the unfiltered pass, not those of earlier rows.
    # The model is trained up front so that both passes read the same training alignments.
    parallel_corpus = _create_two_text_parallel_corpus()
    with _create_trained_model(parallel_corpus) as model:
        corpus = word_align_corpus(parallel_corpus, aligner=model)
        full = list(corpus.get_rows())
        text2_expected = [AlignedWordPair.to_string(r.aligned_word_pairs, include_scores=False) for r in full[2:]]
        text2_actual = _alignment_strings(corpus, ["text2"])
    assert text2_actual == text2_expected


def test_word_align_corpus_transductive_eflomal() -> None:
    rows = list(word_align_corpus(create_test_parallel_corpus(), aligner="eflomal").get_rows())
    assert len(rows) == 8
    assert any(row.aligned_word_pairs for row in rows)


def _create_mock_model(decoy: Decoy) -> ThotSymmetrizedWordAlignmentModel:
    model = decoy.mock(cls=ThotSymmetrizedWordAlignmentModel)
    decoy.when(model.__enter__()).then_return(model)
    decoy.when(model.get_training_alignment(_ANY)).then_return(
        WordAlignmentMatrix.from_word_pairs(row_count=2, column_count=2, set_values=[(0, 0), (1, 1)])
    )
    return model


class _TestEnvironment:
    """Replaces the model that word_align_corpus creates internally with a mock."""

    def __init__(self, decoy: Decoy, monkeypatch: pytest.MonkeyPatch) -> None:
        self.training_corpus: Optional[ParallelTextCorpus] = None

        self.trainer = decoy.mock(cls=Trainer)
        decoy.when(self.trainer.__enter__()).then_return(self.trainer)

        self.model = _create_mock_model(decoy)
        decoy.when(self.model.create_trainer(_ANY)).then_do(self._create_trainer)

        create_model = decoy.mock(func=create_thot_symmetrized_word_alignment_model)
        decoy.when(create_model(_ANY)).then_return(self.model)
        monkeypatch.setattr(thot, "create_thot_symmetrized_word_alignment_model", create_model)

    def _create_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        self.training_corpus = corpus
        return self.trainer


def test_word_align_corpus_trains_lazily(decoy: Decoy, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _TestEnvironment(decoy, monkeypatch)

    corpus = word_align_corpus(create_test_parallel_corpus())
    decoy.verify(env.trainer.train(_ANY), times=0)

    assert len(list(corpus.get_rows())) == 8
    decoy.verify(env.trainer.train(_ANY), times=1)


def test_word_align_corpus_count_does_not_train(decoy: Decoy, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _TestEnvironment(decoy, monkeypatch)

    assert word_align_corpus(create_test_parallel_corpus()).count() == 8
    decoy.verify(env.trainer.train(_ANY), times=0)


def test_word_align_corpus_closes_trained_model(decoy: Decoy, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _TestEnvironment(decoy, monkeypatch)

    with word_align_corpus(create_test_parallel_corpus()).get_rows() as rows:
        assert len(list(rows)) == 8
    # Exiting the model is what closes it, per WordAlignmentModel.__exit__.
    decoy.verify(env.model.__exit__(None, None, None), times=1)


def test_word_align_corpus_closes_trained_model_on_early_exit(decoy: Decoy, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _TestEnvironment(decoy, monkeypatch)

    with word_align_corpus(create_test_parallel_corpus()).get_rows() as rows:
        next(rows)
    decoy.verify(env.model.__exit__(_ANY, _ANY, _ANY), times=1)


def test_word_align_corpus_closes_trained_model_through_chained_operator(
    decoy: Decoy, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cleanup rides on the row generator, so it still happens when a wrapping operator stops early.
    env = _TestEnvironment(decoy, monkeypatch)

    assert len(list(word_align_corpus(create_test_parallel_corpus()).take(3))) == 3
    decoy.verify(env.model.__exit__(_ANY, _ANY, _ANY), times=1)


def test_word_align_corpus_trains_on_requested_text_ids_only(decoy: Decoy, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _TestEnvironment(decoy, monkeypatch)
    corpus = word_align_corpus(_create_two_text_parallel_corpus())

    rows = list(corpus.get_rows(["text2"]))
    assert [row.text_id for row in rows] == ["text2", "text2"]

    assert env.training_corpus is not None
    assert [row.text_id for row in env.training_corpus.get_rows()] == ["text2", "text2"]


def test_word_align_corpus_does_not_close_supplied_model(decoy: Decoy) -> None:
    # A model the caller creates stays the caller's to close, so it can be reused afterward.
    model = _create_mock_model(decoy)

    assert len(list(word_align_corpus(create_test_parallel_corpus(), aligner=model).get_rows())) == 8
    decoy.verify(model.__exit__(_ANY, _ANY, _ANY), times=0)

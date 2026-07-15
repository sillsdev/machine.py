from typing import Iterable, Optional

import pytest
from testutils.thot_test_helpers import create_test_parallel_corpus

from machine.corpora import (
    AlignedWordPair,
    DictionaryTextCorpus,
    MemoryText,
    ParallelTextCorpus,
    StandardParallelTextCorpus,
    TextRow,
)
from machine.translation import SymmetrizationHeuristic, word_align_corpus
from machine.translation.thot import create_thot_symmetrized_word_alignment_model


def _alignment_strings(corpus: ParallelTextCorpus, text_ids: Optional[Iterable[str]] = None) -> list:
    return [
        AlignedWordPair.to_string(row.aligned_word_pairs, include_scores=False) for row in corpus.get_rows(text_ids)
    ]


@pytest.mark.parametrize("aligner", ["fast_align", "ibm1"])
def test_word_align_corpus_transductive_matches_inference(aligner: str) -> None:
    # For deterministic models, the alignments retained during training match those produced by a
    # separate inference pass, so the transductive output must equal aligning each row directly.
    transductive = _alignment_strings(word_align_corpus(create_test_parallel_corpus(), aligner=aligner))

    model = create_thot_symmetrized_word_alignment_model(aligner)
    model.heuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND  # word_align_corpus's default
    with model.create_trainer(create_test_parallel_corpus()) as trainer:
        trainer.train()
        trainer.save()
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
    corpus = word_align_corpus(_create_two_text_parallel_corpus(), aligner="fast_align")
    full = list(corpus.get_rows())
    text2_expected = [AlignedWordPair.to_string(r.aligned_word_pairs, include_scores=False) for r in full[2:]]
    text2_actual = _alignment_strings(corpus, ["text2"])
    assert text2_actual == text2_expected


def test_word_align_corpus_transductive_eflomal() -> None:
    rows = list(word_align_corpus(create_test_parallel_corpus(), aligner="eflomal").get_rows())
    assert len(rows) == 8
    assert any(row.aligned_word_pairs for row in rows)

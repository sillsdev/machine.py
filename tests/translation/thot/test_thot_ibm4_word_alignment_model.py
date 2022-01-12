from typing import Dict, List

from testutils.thot_test_helpers import create_test_parallel_corpus

from machine.translation import WordAlignmentMatrix
from machine.translation.thot import ThotIbm4WordAlignmentModel


def test_create_trainer() -> None:
    model = ThotIbm4WordAlignmentModel()
    model.parameters.ibm1_iteration_count = 2
    model.parameters.hmm_iteration_count = 2
    model.parameters.ibm3_iteration_count = 2
    model.parameters.ibm4_iteration_count = 2
    model.parameters.hmm_p0 = 0.1

    # pronouns
    _add_src_word_class(model, "1", ["isthay", "ouyay", "ityay"])
    # verbs
    _add_src_word_class(model, "2", ["isyay", "ouldshay", "orkway-V", "ancay", "ebay", "esttay-V"])
    # articles
    _add_src_word_class(model, "3", ["ayay"])
    # nouns
    _add_src_word_class(model, "4", ["esttay-N", "orkway-N", "ordway"])
    # punctuation
    _add_src_word_class(model, "5", [".", "?", "!"])
    # adverbs
    _add_src_word_class(model, "6", ["oftenyay"])
    # adjectives
    _add_src_word_class(model, "7", ["ardhay", "orkingway"])

    # pronouns
    _add_trg_word_class(model, "1", ["this", "you", "it"])
    # verbs
    _add_trg_word_class(model, "2", ["is", "should", "can", "be"])
    # articles
    _add_trg_word_class(model, "3", ["a"])
    # nouns
    _add_trg_word_class(model, "4", ["word"])
    # punctuation
    _add_trg_word_class(model, "5", [".", "?", "!"])
    # adverbs
    _add_trg_word_class(model, "6", ["often"])
    # adjectives
    _add_trg_word_class(model, "7", ["hard", "working"])
    # nouns/verbs
    _add_trg_word_class(model, "8", ["test", "work"])
    # disambiguators
    _add_trg_word_class(model, "9", ["N", "V"])

    trainer = model.create_trainer(create_test_parallel_corpus())
    trainer.train()
    trainer.save()

    matrix = model.get_best_alignment("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())
    assert matrix == WordAlignmentMatrix.from_word_pairs(5, 6, {(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5)})

    matrix = model.get_best_alignment("isthay isyay otnay ayay esttay-N .".split(), "this is not a test N .".split())
    assert matrix == WordAlignmentMatrix.from_word_pairs(6, 7, {(0, 0), (1, 1), (3, 3), (4, 4), (4, 5), (5, 6)})

    matrix = model.get_best_alignment("isthay isyay ayay esttay-N ardhay .".split(), "this is a hard test N .".split())
    assert matrix == WordAlignmentMatrix.from_word_pairs(6, 7, {(0, 0), (1, 1), (2, 2), (4, 3), (3, 4), (3, 5), (5, 6)})


def _add_src_word_class(model: ThotIbm4WordAlignmentModel, word_class: str, words: List[str]) -> None:
    _add_word_class(model.parameters.source_word_classes, word_class, words)


def _add_trg_word_class(model: ThotIbm4WordAlignmentModel, word_class: str, words: List[str]) -> None:
    _add_word_class(model.parameters.target_word_classes, word_class, words)


def _add_word_class(word_classes: Dict[str, str], word_class: str, words: List[str]) -> None:
    for word in words:
        word_classes[word] = word_class

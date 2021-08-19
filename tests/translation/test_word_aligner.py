from mockito import ANY, when
from test_utils import make_concrete

from machine.translation import WordAligner, WordAlignmentMatrix


def test_get_best_alignment_from_known() -> None:
    estimated_alignment = WordAlignmentMatrix(10, 7, {(1, 1), (2, 1), (4, 2), (5, 1), (6, 3), (7, 4), (8, 5), (9, 6)})
    TestWordAligner = make_concrete(WordAligner)
    when(TestWordAligner).get_best_alignment(ANY, ANY).thenReturn(estimated_alignment)
    aligner = TestWordAligner()
    known_alignment = WordAlignmentMatrix(10, 7, {(0, 0), (6, 3), (7, 5), (8, 4)})
    alignment = aligner.get_best_alignment_from_known(
        "maria no daba una bofetada a la bruja verde .".split(),
        "mary didn't slap the green witch .".split(),
        known_alignment,
    )
    assert alignment == WordAlignmentMatrix(10, 7, {(0, 0), (1, 1), (2, 1), (4, 2), (6, 3), (8, 4), (7, 5), (9, 6)})

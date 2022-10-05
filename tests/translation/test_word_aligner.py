from mockito import ANY, when
from testutils import make_concrete

from machine.corpora.parallel_text_row import ParallelTextRow
from machine.translation import WordAligner, WordAlignmentMatrix


def test_align_parallel_text_row() -> None:
    known_alignment = WordAlignmentMatrix.from_word_pairs(10, 7, {(0, 0), (6, 3), (7, 5), (8, 4)})
    row = ParallelTextRow(
        "text1",
        ["1"],
        ["2"],
        "maria no daba una bofetada a la bruja verde .".split(),
        "mary didn't slap the green witch .".split(),
        aligned_word_pairs=known_alignment.to_aligned_word_pairs(),
    )
    estimated_alignment = WordAlignmentMatrix.from_word_pairs(
        10, 7, {(1, 1), (2, 1), (4, 2), (5, 1), (6, 3), (7, 4), (8, 5), (9, 6)}
    )
    TestWordAligner = make_concrete(WordAligner)
    when(TestWordAligner).align(ANY, ANY).thenReturn(estimated_alignment)
    aligner = TestWordAligner()  # type: ignore
    alignment = aligner.align_parallel_text_row(row)
    assert alignment == WordAlignmentMatrix.from_word_pairs(
        10, 7, {(0, 0), (1, 1), (2, 1), (4, 2), (6, 3), (8, 4), (7, 5), (9, 6)}
    )

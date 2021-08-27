from typing import Tuple

from machine.translation import WordAlignmentMatrix


def test_intersect_with() -> None:
    x, y = _create_matrices()
    x.intersect_with(y)
    assert x == WordAlignmentMatrix.from_word_pairs(7, 9, {(0, 0), (2, 1), (3, 4)})


def test_union_with() -> None:
    x, y = _create_matrices()
    x.union_with(y)
    assert x == WordAlignmentMatrix.from_word_pairs(
        7, 9, {(0, 0), (1, 1), (1, 5), (2, 1), (3, 2), (3, 3), (3, 4), (4, 5), (4, 6), (5, 3), (6, 8)}
    )


def test_symmetrize_with() -> None:
    x, y = _create_matrices()
    x.symmetrize_with(y)
    assert x == WordAlignmentMatrix.from_word_pairs(
        7, 9, {(0, 0), (1, 1), (2, 1), (3, 2), (3, 3), (3, 4), (4, 5), (4, 6), (6, 8)}
    )


def test_grow_symmetrize_with() -> None:
    x, y = _create_matrices()
    x.grow_symmetrize_with(y)
    assert x == WordAlignmentMatrix.from_word_pairs(7, 9, {(0, 0), (1, 1), (2, 1), (3, 2), (3, 3), (3, 4)})


def test_grow_diag_symmetrize_with() -> None:
    x, y = _create_matrices()
    x.grow_diag_symmetrize_with(y)
    assert x == WordAlignmentMatrix.from_word_pairs(
        7, 9, {(0, 0), (1, 1), (2, 1), (3, 2), (3, 3), (3, 4), (4, 5), (4, 6)}
    )


def test_grow_diag_final_symmetrize_with() -> None:
    x, y = _create_matrices()
    x.grow_diag_final_symmetrize_with(y)
    assert x == WordAlignmentMatrix.from_word_pairs(
        7, 9, {(0, 0), (1, 1), (2, 1), (3, 2), (3, 3), (3, 4), (4, 5), (4, 6), (5, 3), (6, 8)}
    )


def test_grow_diag_final_and_symmetrize_with() -> None:
    x, y = _create_matrices()
    x.grow_diag_final_and_symmetrize_with(y)
    assert x == WordAlignmentMatrix.from_word_pairs(
        7, 9, {(0, 0), (1, 1), (2, 1), (3, 2), (3, 3), (3, 4), (4, 5), (4, 6), (6, 8)}
    )


def test_resize_grow() -> None:
    matrix = WordAlignmentMatrix.from_word_pairs(3, 3, {(0, 0), (1, 1), (2, 2)})
    matrix.resize(4, 4)
    assert matrix == WordAlignmentMatrix.from_word_pairs(4, 4, {(0, 0), (1, 1), (2, 2)})


def test_resize_shrink() -> None:
    matrix = WordAlignmentMatrix.from_word_pairs(3, 3, {(0, 0), (1, 1), (2, 2)})
    matrix.resize(2, 2)
    assert matrix == WordAlignmentMatrix.from_word_pairs(2, 2, {(0, 0), (1, 1)})


def test_resize_grow_and_shrink() -> None:
    matrix = WordAlignmentMatrix.from_word_pairs(3, 3, {(0, 0), (1, 1), (2, 2)})
    matrix.resize(2, 4)
    assert matrix == WordAlignmentMatrix.from_word_pairs(2, 4, {(0, 0), (1, 1)})


def _create_matrices() -> Tuple[WordAlignmentMatrix, WordAlignmentMatrix]:
    x = WordAlignmentMatrix.from_word_pairs(7, 9, {(0, 0), (1, 5), (2, 1), (3, 2), (3, 3), (3, 4), (4, 5), (5, 3)})
    y = WordAlignmentMatrix.from_word_pairs(7, 9, {(0, 0), (1, 1), (2, 1), (3, 4), (4, 6), (6, 8)})
    return x, y

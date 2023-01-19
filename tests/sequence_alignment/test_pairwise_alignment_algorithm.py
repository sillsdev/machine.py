from typing import Iterable, List, Optional, Tuple

from pytest import approx

from machine.sequence_alignment import (
    Alignment,
    AlignmentCell,
    AlignmentMode,
    PairwiseAlignmentAlgorithm,
    PairwiseAlignmentScorer,
)


class _StringScorer(PairwiseAlignmentScorer[str, str]):
    def get_gap_penalty(self, sequence1: str, sequence2: str) -> int:
        return -100

    def get_insertion_score(self, sequence1: str, p: Optional[str], sequence2: str, q: str) -> int:
        return 0

    def get_deletion_score(self, sequence1: str, p: str, sequence2: str, q: Optional[str]) -> int:
        return 0

    def get_substitution_score(self, sequence1: str, p: str, sequence2: str, q: str) -> int:
        return 100 if p == q else 0

    def get_expansion_score(self, sequence1: str, p: str, sequence2: str, q1: str, q2: str) -> int:
        score = 0
        if p == q1:
            score += 100
        if p == q2:
            score += 100
        return score

    def get_compression_score(self, sequence1: str, p1: str, p2: str, sequence2: str, q: str) -> int:
        score = 0
        if q == p1:
            score += 100
        if q == p2:
            score += 100
        return score

    def get_transposition_score(self, sequence1: str, p1: str, p2: str, sequence2: str, q1: str, q2: str) -> int:
        return 100 if p1 == q2 and p2 == q2 else 0

    def get_max_score1(self, sequence1: str, p: str, sequence2: str) -> int:
        return 100

    def get_max_score2(self, sequence1: str, sequence2: str, q: str) -> int:
        return 100


class _ZeroMaxStringScorer(_StringScorer):
    def get_max_score1(self, sequence1: str, p: str, sequence2: str) -> int:
        return 0

    def get_max_score2(self, sequence1: str, sequence2: str, q: str) -> int:
        return 0


def _get_chars(sequence: str) -> Tuple[Iterable[str], int, int]:
    return sequence, 0, len(sequence)


def _create_alignment(*alignment: str) -> Alignment[str, str]:
    sequences: List[Tuple[str, AlignmentCell[str], Iterable[AlignmentCell[str]], AlignmentCell[str]]] = []
    for i in range(len(alignment)):
        seq = ""
        split = alignment[i].split("|")
        prefix = split[0].strip()
        seq += prefix

        cell_strs = split[1].strip().split()
        cells: List[AlignmentCell[str]] = []
        for cell_str in cell_strs:
            if cell_str == "-":
                cells.append(AlignmentCell[str]())
            else:
                seq += cell_str
                cells.append(AlignmentCell(cell_str))

        suffix = split[2].strip()
        seq += suffix

        sequences.append((seq, AlignmentCell(prefix), cells, AlignmentCell(suffix)))
    return Alignment(0, 0, sequences)


def _assert_alignments_equal(actual: Alignment[str, str], expected: Alignment[str, str]) -> None:
    assert actual.sequences == expected.sequences
    assert actual.prefixes == expected.prefixes
    assert actual.suffixes == expected.suffixes
    assert actual.column_count == expected.column_count
    for i in range(expected.sequence_count):
        for j in range(expected.column_count):
            assert actual[i, j] == expected[i, j]


def test_global_align() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "car", "bar", _get_chars)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r |", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.66, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "bar", _get_chars)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r t |", "| b a r - |"))
    assert alignments[0].normalized_score == approx(0.25, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "art", _get_chars)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r t |", "| - a r t |"))
    assert alignments[0].normalized_score == approx(0.5, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "start", "tan", _get_chars)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 2
    _assert_alignments_equal(alignments[0], _create_alignment("| s t a r t |", "| - t a - n |"))
    assert alignments[0].normalized_score == approx(0.0, abs=0.01)
    _assert_alignments_equal(alignments[1], _create_alignment("| s t a r t |", "| - t a n - |"))
    assert alignments[1].normalized_score == approx(0.0, abs=0.01)


def test_local_align() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "car", "bar", _get_chars, mode=AlignmentMode.LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 2
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r |", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.66, abs=0.01)
    _assert_alignments_equal(alignments[1], _create_alignment("c | a r |", "b | a r |"))
    assert alignments[1].normalized_score == approx(0.8, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "bar", _get_chars, mode=AlignmentMode.LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 2
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r | t", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.57, abs=0.01)
    _assert_alignments_equal(alignments[1], _create_alignment("c | a r | t", "b | a r |"))
    assert alignments[1].normalized_score == approx(0.66, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "art", _get_chars, mode=AlignmentMode.LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("c | a r t |", "| a r t |"))
    assert alignments[0].normalized_score == approx(0.86, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "start", "tan", _get_chars, mode=AlignmentMode.LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 2
    _assert_alignments_equal(alignments[0], _create_alignment("s | t a | rt", "| t a | n"))
    assert alignments[0].normalized_score == approx(0.57, abs=0.01)
    _assert_alignments_equal(alignments[1], _create_alignment("s | t a r | t", "| t a n |"))
    assert alignments[1].normalized_score == approx(0.5, abs=0.01)


def test_half_local_align() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "car", "bar", _get_chars, mode=AlignmentMode.HALF_LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r |", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.66, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "bar", _get_chars, mode=AlignmentMode.HALF_LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r | t", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.57, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "art", _get_chars, mode=AlignmentMode.HALF_LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r t |", "| - a r t |"))
    assert alignments[0].normalized_score == approx(0.5, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "start", "tan", _get_chars, mode=AlignmentMode.HALF_LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 2
    _assert_alignments_equal(alignments[0], _create_alignment("| s t a | rt", "| - t a | n"))
    assert alignments[0].normalized_score == approx(0.25, abs=0.01)
    _assert_alignments_equal(alignments[1], _create_alignment("| s t a r | t", "| - t a n |"))
    assert alignments[1].normalized_score == approx(0.22, abs=0.01)


def test_semi_global_align() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "car", "bar", _get_chars, mode=AlignmentMode.SEMI_GLOBAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r |", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.66, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "bar", _get_chars, mode=AlignmentMode.SEMI_GLOBAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r | t", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.57, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "art", _get_chars, mode=AlignmentMode.SEMI_GLOBAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("c | a r t |", "| a r t |"))
    assert alignments[0].normalized_score == approx(0.86, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "start", "tan", _get_chars, mode=AlignmentMode.SEMI_GLOBAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("s | t a r | t", "| t a n |"))
    assert alignments[0].normalized_score == approx(0.5, abs=0.01)


def test_expansion_compression_align() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "car", "bar", _get_chars, expansion_compression_enabled=True)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r |", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.66, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "bar", _get_chars, expansion_compression_enabled=True)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a rt |", "| b a r |"))
    assert alignments[0].normalized_score == approx(0.5, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "cart", "art", _get_chars, expansion_compression_enabled=True)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| ca r t |", "| a r t |"))
    assert alignments[0].normalized_score == approx(0.75, abs=0.01)

    paa = PairwiseAlignmentAlgorithm(scorer, "start", "tan", _get_chars, expansion_compression_enabled=True)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 2
    _assert_alignments_equal(alignments[0], _create_alignment("| st ar t |", "| t  a  n |"))
    assert alignments[0].normalized_score == approx(0.4, abs=0.01)
    _assert_alignments_equal(alignments[1], _create_alignment("| st a rt |", "| t  a n |"))
    assert alignments[1].normalized_score == approx(0.4, abs=0.01)


def test_zero_max_score() -> None:
    scorer = _ZeroMaxStringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "car", "bar", _get_chars, expansion_compression_enabled=True)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("| c a r |", "| b a r |"))
    assert alignments[0].normalized_score == 0


def test_global_align_empty_sequence() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "", "", _get_chars)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("||", "||"))
    assert alignments[0].normalized_score == 0


def test_local_align_empty_sequence() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "", "", _get_chars, mode=AlignmentMode.LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("||", "||"))
    assert alignments[0].normalized_score == 0


def test_half_local_align_empty_sequence() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "", "", _get_chars, mode=AlignmentMode.HALF_LOCAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("||", "||"))
    assert alignments[0].normalized_score == 0


def test_semi_global_align_empty_sequence() -> None:
    scorer = _StringScorer()
    paa = PairwiseAlignmentAlgorithm(scorer, "", "", _get_chars, mode=AlignmentMode.SEMI_GLOBAL)
    paa.compute()
    alignments = list(paa.get_alignments())

    assert len(alignments) == 1
    _assert_alignments_equal(alignments[0], _create_alignment("||", "||"))
    assert alignments[0].normalized_score == 0

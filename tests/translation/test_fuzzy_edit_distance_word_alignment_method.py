from typing import Sequence

from machine.translation import FuzzyEditDistanceWordAlignmentMethod


def test_align_last_src_first_trg() -> None:
    def score_selector(src_segment: Sequence[str], src_idx: int, trg_segment: Sequence[str], trg_idx: int) -> float:
        if src_idx == -1 or trg_idx == -1:
            return 0.1
        return 0.9 if src_segment[src_idx] == trg_segment[trg_idx] else 0.1

    method = FuzzyEditDistanceWordAlignmentMethod(score_selector=score_selector)

    matrix = method.align("A B".split(), "B C".split())
    assert str(matrix) == "1-0"

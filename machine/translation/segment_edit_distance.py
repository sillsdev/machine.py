from typing import Iterable, List, Sequence, Tuple

from .edit_distance import EditDistance
from .edit_operation import EditOperation
from .word_edit_distance import WordEditDistance


class SegmentEditDistance(EditDistance[Sequence[str], str]):
    def __init__(self) -> None:
        self._word_edit_distance = WordEditDistance()

    @property
    def hit_cost(self) -> float:
        return self._word_edit_distance.hit_cost

    @hit_cost.setter
    def hit_cost(self, cost: float) -> None:
        self._word_edit_distance.hit_cost = cost

    @property
    def substitution_cost(self) -> float:
        return self._word_edit_distance.substitution_cost

    @substitution_cost.setter
    def substitution_cost(self, cost: float) -> None:
        self._word_edit_distance.substitution_cost = cost

    @property
    def insertion_cost(self) -> float:
        return self._word_edit_distance.insertion_cost

    @insertion_cost.setter
    def insertion_cost(self, cost: float) -> None:
        self._word_edit_distance.insertion_cost = cost

    @property
    def deletion_cost(self) -> float:
        return self._word_edit_distance.deletion_cost

    @deletion_cost.setter
    def deletion_cost(self, cost: float) -> None:
        self._word_edit_distance.deletion_cost = cost

    def compute(self, x: Sequence[str], y: Sequence[str]) -> float:
        dist, _ = self._compute_dist_matrix(x, y, is_last_item_complete=True, use_prefix_del_op=False)
        return dist

    def compute_prefix(
        self, x: Sequence[str], y: Sequence[str], is_last_item_complete: bool, use_prefix_del_op: bool
    ) -> Tuple[float, Iterable[EditOperation], Iterable[EditOperation]]:
        dist, dist_matrix = self._compute_dist_matrix(x, y, is_last_item_complete, use_prefix_del_op)

        i = len(x)
        j = len(y)
        ops: List[EditOperation] = []
        char_ops: Iterable[EditOperation] = []
        while i > 0 or j > 0:
            _, i, j, op = self._process_dist_matrix_cell(
                x, y, dist_matrix, use_prefix_del_op, j != len(y) or is_last_item_complete, i, j
            )
            if op != EditOperation.PREFIX_DELETE:
                ops.append(op)

            if j + 1 == len(y) and not is_last_item_complete and op == EditOperation.HIT:
                _, char_ops = self._word_edit_distance.compute_prefix(
                    x[i], y[-1], is_last_item_complete=True, use_prefix_del_op=True
                )

        ops.reverse()
        return (dist, ops, char_ops)

    def incr_compute_prefix_first_row(
        self, scores: List[float], prev_scores: List[float], y_incr: Sequence[str]
    ) -> None:
        if scores is not prev_scores:
            scores.clear()
            scores.extend(prev_scores)

        start_pos = len(scores)
        for j_incr in range(len(y_incr)):
            j = start_pos + j_incr
            if j == 0:
                scores.append(self._get_insertion_cost(y_incr[j_incr]))
            else:
                scores.append(scores[j - 1] + self._get_insertion_cost(y_incr[j_incr]))

    def incr_compute_prefix(
        self,
        scores: List[float],
        prev_scores: List[float],
        x_word: str,
        y_incr: Sequence[str],
        is_last_item_complete: bool,
    ) -> Iterable[EditOperation]:
        x = [x_word]
        y = [""] * (len(prev_scores) - 1)
        for i in range(len(y_incr)):
            y[len(prev_scores) - len(y_incr) - 1 + i] = y_incr[i]

        dist_matrix = self._init_dist_matrix(x, y)

        for j in range(len(prev_scores)):
            dist_matrix[0][j] = prev_scores[j]
        for j in range(len(scores)):
            dist_matrix[1][j] = scores[j]

        while len(scores) < len(prev_scores):
            scores.append(0.0)

        start_pos = len(prev_scores) - len(y_incr)

        ops: List[EditOperation] = []
        for j_incr in range(len(y_incr)):
            j = start_pos + j_incr
            dist, _, _, op = self._process_dist_matrix_cell(
                x,
                y,
                dist_matrix,
                use_prefix_del_op=False,
                is_complete=j != len(y) or is_last_item_complete,
                i=1,
                j=j,
            )
            scores[j] = dist
            dist_matrix[1][j] = dist
            ops.append(op)

        return ops

    def _get_count(self, seq: Sequence[str]) -> int:
        return len(seq)

    def _get_item(self, seq: Sequence[str], index: int) -> str:
        return seq[index]

    def _get_hit_cost(self, x: str, y: str, is_complete: bool) -> float:
        return self.hit_cost * len(y)

    def _get_substitution_cost(self, x: str, y: str, is_complete: bool) -> float:
        if x == "":
            return (self.substitution_cost * 0.99) * len(y)

        if is_complete:
            _, ops = self._word_edit_distance.compute(x, y)
        else:
            _, ops = self._word_edit_distance.compute_prefix(x, y, is_last_item_complete=True, use_prefix_del_op=True)

        hit_count, ins_count, subst_count, del_count = _get_op_counts(ops)
        return (
            self.hit_cost * hit_count
            + self.insertion_cost * ins_count
            + self.substitution_cost * subst_count
            + self.deletion_cost * del_count
        )

    def _get_deletion_cost(self, x: str) -> float:
        if x == "":
            return self.deletion_cost
        return self.deletion_cost * len(x)

    def _get_insertion_cost(self, y: str) -> float:
        return self.insertion_cost * len(y)

    def _is_hit(self, x: str, y: str, is_complete: bool) -> bool:
        return x == y or (not is_complete and x.startswith(y))


def _get_op_counts(ops: Iterable[EditOperation]) -> Tuple[int, int, int, int]:
    hit_count = 0
    ins_count = 0
    subst_count = 0
    del_count = 0
    for op in ops:
        if op == EditOperation.HIT:
            hit_count += 1
        elif op == EditOperation.INSERT:
            ins_count += 1
        elif op == EditOperation.SUBSTITUTE:
            subst_count += 1
        elif op == EditOperation.DELETE:
            del_count += 1
    return (hit_count, ins_count, subst_count, del_count)

from typing import Iterable, Tuple

from .edit_distance import EditDistance
from .edit_operation import EditOperation


class WordEditDistance(EditDistance[str, str]):
    def __init__(self) -> None:
        self.hit_cost = 0.0
        self.insertion_cost = 0.0
        self.deletion_cost = 0.0
        self.substitution_cost = 0.0

    def compute(self, x: str, y: str) -> Tuple[float, Iterable[EditOperation]]:
        dist, dist_matrix = self._compute_dist_matrix(x, y, is_last_item_complete=True, use_prefix_del_op=False)
        ops = self._get_operations(
            x,
            y,
            dist_matrix,
            is_last_item_complete=True,
            use_prefix_del_op=False,
            i=self._get_count(x),
            j=self._get_count(y),
        )
        return (dist, ops)

    def compute_prefix(
        self, x: str, y: str, is_last_item_complete: bool, use_prefix_del_op: bool
    ) -> Tuple[float, Iterable[EditOperation]]:
        dist, dist_matrix = self._compute_dist_matrix(x, y, is_last_item_complete, use_prefix_del_op)
        ops = self._get_operations(
            x,
            y,
            dist_matrix,
            is_last_item_complete,
            use_prefix_del_op,
            i=self._get_count(x),
            j=self._get_count(y),
        )
        return (dist, ops)

    def _get_count(self, seq: str) -> int:
        return len(seq)

    def _get_item(self, seq: str, index: int) -> str:
        return seq[index]

    def _get_hit_cost(self, x: str, y: str, is_complete: bool) -> float:
        return self.hit_cost

    def _get_substitution_cost(self, x: str, y: str, is_complete: bool) -> float:
        return self.substitution_cost

    def _get_deletion_cost(self, x: str) -> float:
        return self.deletion_cost

    def _get_insertion_cost(self, y: str) -> float:
        return self.insertion_cost

    def _is_hit(self, x: str, y: str, is_complete: bool) -> bool:
        return x == y

from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Tuple, TypeVar

from .edit_operation import EditOperation

Seq = TypeVar("Seq")
Item = TypeVar("Item")


class EditDistance(ABC, Generic[Seq, Item]):
    @abstractmethod
    def _get_count(self, seq: Seq) -> int: ...

    @abstractmethod
    def _get_item(self, seq: Seq, index: int) -> Item: ...

    @abstractmethod
    def _get_hit_cost(self, x: Item, y: Item, is_complete: bool) -> float: ...

    @abstractmethod
    def _get_substitution_cost(self, x: Item, y: Item, is_complete: bool) -> float: ...

    @abstractmethod
    def _get_deletion_cost(self, x: Item) -> float: ...

    @abstractmethod
    def _get_insertion_cost(self, y: Item) -> float: ...

    @abstractmethod
    def _is_hit(self, x: Item, y: Item, is_complete: bool) -> bool: ...

    def _init_dist_matrix(self, x: Seq, y: Seq) -> List[List[float]]:
        x_count = self._get_count(x)
        y_count = self._get_count(y)
        dim = max(x_count, y_count)
        dist_matrix = [[0.0 for _ in range(dim + 1)] for _ in range(dim + 1)]
        return dist_matrix

    def _compute_dist_matrix(
        self, x: Seq, y: Seq, is_last_item_complete: bool, use_prefix_del_op: bool
    ) -> Tuple[float, List[List[float]]]:
        dist_matrix = self._init_dist_matrix(x, y)

        x_count = self._get_count(x)
        y_count = self._get_count(y)
        for i in range(x_count + 1):
            for j in range(y_count + 1):
                dist_matrix[i][j], _, _, _ = self._process_dist_matrix_cell(
                    x, y, dist_matrix, use_prefix_del_op, j != y_count or is_last_item_complete, i, j
                )

        return dist_matrix[x_count][y_count], dist_matrix

    def _process_dist_matrix_cell(
        self,
        x: Seq,
        y: Seq,
        dist_matrix: List[List[float]],
        use_prefix_del_op: bool,
        is_complete: bool,
        i: int,
        j: int,
    ) -> Tuple[float, int, int, EditOperation]:
        if i != 0 and j != 0:
            x_item = self._get_item(x, i - 1)
            y_item = self._get_item(y, j - 1)
            if self._is_hit(x_item, y_item, is_complete):
                subst_cost = self._get_hit_cost(x_item, y_item, is_complete)
                op = EditOperation.HIT
            else:
                subst_cost = self._get_substitution_cost(x_item, y_item, is_complete)
                op = EditOperation.SUBSTITUTE

            cost = dist_matrix[i - 1][j - 1] + subst_cost
            min = cost
            i_pred = i - 1
            j_pred = j - 1

            del_cost = 0 if use_prefix_del_op and j == self._get_count(y) else self._get_deletion_cost(x_item)
            cost = dist_matrix[i - 1][j] + del_cost
            if cost < min:
                min = cost
                i_pred = i - 1
                j_pred = j
                op = EditOperation.PREFIX_DELETE if del_cost == 0 else EditOperation.DELETE

            cost = dist_matrix[i][j - 1] + self._get_insertion_cost(y_item)
            if cost < min:
                min = cost
                i_pred = i
                j_pred = j - 1
                op = EditOperation.INSERT

            return (min, i_pred, j_pred, op)

        if i == 0 and j == 0:
            return (0.0, 0, 0, EditOperation.NONE)

        if i == 0:
            return (
                dist_matrix[0][j - 1] + self._get_insertion_cost(self._get_item(y, j - 1)),
                0,
                j - 1,
                EditOperation.INSERT,
            )

        return (
            dist_matrix[i - 1][0] + self._get_deletion_cost(self._get_item(x, i - 1)),
            i - 1,
            0,
            EditOperation.DELETE,
        )

    def _get_operations(
        self,
        x: Seq,
        y: Seq,
        dist_matrix: List[List[float]],
        is_last_item_complete: bool,
        use_prefix_del_op: bool,
        i: int,
        j: int,
    ) -> Iterable[EditOperation]:
        y_count = self._get_count(y)
        ops: List[EditOperation] = []
        while i > 0 or j > 0:
            _, i, j, op = self._process_dist_matrix_cell(
                x, y, dist_matrix, use_prefix_del_op, j != y_count or is_last_item_complete, i, j
            )
            if op != EditOperation.PREFIX_DELETE:
                ops.append(op)
        ops.reverse()
        return ops

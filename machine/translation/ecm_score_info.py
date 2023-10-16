from __future__ import annotations

from typing import List

from .edit_operation import EditOperation


class EcmScoreInfo:
    def __init__(self) -> None:
        self._scores: List[float] = []
        self._operations: List[EditOperation] = []

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def operations(self) -> List[EditOperation]:
        return self._operations

    def update_positions(self, prev_esi: EcmScoreInfo, positions: List[int]) -> None:
        while len(self.scores) < len(prev_esi.scores):
            self.scores.append(0.0)

        while len(self.operations) < len(prev_esi.operations):
            self.operations.append(EditOperation.NONE)

        for i in range(len(positions)):
            self.scores[positions[i]] = prev_esi.scores[positions[i]]
            if len(prev_esi.operations) > i:
                self.operations[positions[i]] = prev_esi.operations[positions[i]]

    def remove_last(self) -> None:
        if len(self.scores) > 1:
            self.scores.pop()
        if len(self.operations) > 1:
            self.operations.pop()

    def get_last_ins_prefix_word_from_esi(self) -> List[int]:
        results = [0] * len(self.operations)

        for j in range(len(self.operations) - 1, -1, -1):
            if self.operations[j] == EditOperation.HIT:
                results[j] = j - 1
            elif self.operations[j] == EditOperation.INSERT:
                tj = j
                while tj >= 0 and self.operations[tj] == EditOperation.INSERT:
                    tj -= 1
                if self.operations[tj] == EditOperation.HIT or self.operations[tj] == EditOperation.SUBSTITUTE:
                    tj -= 1
                results[j] = tj
            elif self.operations[j] == EditOperation.DELETE:
                results[j] = j
            elif self.operations[j] == EditOperation.SUBSTITUTE:
                results[j] = j - 1
            elif self.operations[j] == EditOperation.NONE:
                results[j] = 0

        return results

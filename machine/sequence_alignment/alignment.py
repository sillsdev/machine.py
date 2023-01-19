from math import isnan
from typing import Generic, Iterable, List, Sequence, Tuple, TypeVar

from .alignment_cell import AlignmentCell

Seq = TypeVar("Seq")
Item = TypeVar("Item")


class Alignment(Generic[Seq, Item]):
    def __init__(
        self,
        raw_score: int,
        normalized_score: float,
        sequences: Iterable[Tuple[Seq, AlignmentCell[Item], Iterable[AlignmentCell[Item]], AlignmentCell[Item]]],
    ) -> None:
        if isnan(normalized_score) or normalized_score < 0 or normalized_score > 1:
            raise ValueError("normalized_score is out of range.")

        self._raw_score = raw_score
        self._normalized_score = normalized_score
        self._sequences: List[Seq] = []
        self._prefixes: List[AlignmentCell[Item]] = []
        self._matrix: List[List[AlignmentCell[Item]]] = []
        self._suffixes: List[AlignmentCell[Item]] = []
        self._column_count = -1
        for seq, prefix, columns, suffix in sequences:
            self._sequences.append(seq)
            self._prefixes.append(prefix)
            column_list = list(columns)
            if self._column_count != -1 and self._column_count != len(column_list):
                raise ValueError("All sequences are not the same length.")
            self._column_count = len(column_list)
            self._matrix.append(column_list)
            self._suffixes.append(suffix)

        if len(self._sequences) == 0:
            raise ValueError("No sequences specified.")

    @property
    def raw_score(self) -> int:
        return self._raw_score

    @property
    def normalized_score(self) -> float:
        return self._normalized_score

    @property
    def sequence_count(self) -> int:
        return len(self._sequences)

    @property
    def column_count(self) -> int:
        return self._column_count

    @property
    def sequences(self) -> Sequence[Seq]:
        return self._sequences

    @property
    def prefixes(self) -> Sequence[AlignmentCell[Item]]:
        return self._prefixes

    @property
    def suffixes(self) -> Sequence[AlignmentCell[Item]]:
        return self._suffixes

    def __getitem__(self, key: Tuple[int, int]) -> AlignmentCell[Item]:
        return self._matrix[key[0]][key[1]]

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

Seq = TypeVar("Seq")
Item = TypeVar("Item")


class PairwiseAlignmentScorer(ABC, Generic[Seq, Item]):
    @abstractmethod
    def get_gap_penalty(self, sequence1: Seq, sequence2: Seq) -> int:
        ...

    @abstractmethod
    def get_insertion_score(self, sequence1: Seq, p: Optional[Item], sequence2: Seq, q: Item) -> int:
        ...

    @abstractmethod
    def get_deletion_score(self, sequence1: Seq, p: Item, sequence2: Seq, q: Optional[Item]) -> int:
        ...

    @abstractmethod
    def get_substitution_score(self, sequence1: Seq, p: Item, sequence2: Seq, q: Item) -> int:
        ...

    @abstractmethod
    def get_expansion_score(self, sequence1: Seq, p: Item, sequence2: Seq, q1: Item, q2: Item) -> int:
        ...

    @abstractmethod
    def get_compression_score(self, sequence1: Seq, p1: Item, p2: Item, sequence2: Seq, q: Item) -> int:
        ...

    @abstractmethod
    def get_transposition_score(
        self,
        sequence1: Seq,
        p1: Item,
        p2: Item,
        sequence2: Seq,
        q1: Item,
        q2: Item,
    ) -> int:
        ...

    @abstractmethod
    def get_max_score1(self, sequence1: Seq, p: Item, sequence2: Seq) -> int:
        ...

    @abstractmethod
    def get_max_score2(self, sequence1: Seq, sequence2: Seq, q: Item) -> int:
        ...

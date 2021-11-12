from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

from networkx import Graph

from .cluster import Cluster

T = TypeVar("T")


class UnrootedHierarchicalClusterer(ABC, Generic[T]):
    @abstractmethod
    def generate_clusters(self, data_objects: Iterable[T]) -> Graph[Cluster[T]]:
        ...

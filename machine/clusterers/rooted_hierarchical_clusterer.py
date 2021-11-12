from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

from networkx import DiGraph

from .cluster import Cluster

T = TypeVar("T")


class RootedHierarchicalClusterer(ABC, Generic[T]):
    @abstractmethod
    def generate_clusters(self, data_objects: Iterable[T]) -> DiGraph[Cluster[T]]:
        ...

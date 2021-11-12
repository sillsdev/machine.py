from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

from .cluster import Cluster

T = TypeVar("T")


class FlatClusterer(ABC, Generic[T]):
    @abstractmethod
    def generate_clusters(self, data_objects: Iterable[T]) -> Iterable[Cluster[T]]:
        ...

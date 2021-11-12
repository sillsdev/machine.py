import sys
from itertools import chain
from math import sqrt
from statistics import mean
from typing import Callable, Iterable, List, TypeVar

from .cluster import Cluster
from .flat_clusterer import FlatClusterer

T = TypeVar("T")


class FlatUpgmaClusterer(FlatClusterer[T]):
    def __init__(self, get_distance: Callable[[T, T], float], threshold: float) -> None:
        self._get_distance = get_distance
        self._threshold = threshold

    def generate_clusters(self, data_objects: Iterable[T]) -> Iterable[Cluster[T]]:
        clusters: List[Cluster[T]] = [
            Cluster[T](data_object, description=str(data_object)) for data_object in data_objects
        ]
        while len(clusters) >= 2:
            min_i = 0
            min_j = 0
            min_score = sys.float_info.max
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dists: List[float] = []
                    for o1 in clusters[i].data_objects:
                        for o2 in clusters[j].data_objects:
                            dists.append(self._get_distance(o1, o2))
                    avg = mean(dists)
                    total = sum((d - avg) * (d - avg) for d in dists)
                    std_dev = sqrt(total / len(dists))
                    score = avg - 0.25 * std_dev
                    if score < min_score:
                        min_score = score
                        min_i = i
                        min_j = j

            if min_score > self._threshold:
                break

            i_cluster = clusters[min_i]
            j_cluster = clusters[min_j]
            u_cluster = Cluster[T](chain(i_cluster.data_objects, j_cluster.data_objects))
            del clusters[min_j]
            del clusters[min_i]
            clusters.append(u_cluster)

        return clusters

from __future__ import annotations

import sys
from typing import Callable, Dict, FrozenSet, Iterable, List, TypeVar

from networkx import DiGraph

from .cluster import Cluster
from .rooted_hierarchical_clusterer import RootedHierarchicalClusterer

T = TypeVar("T")


class UpgmaClusterer(RootedHierarchicalClusterer[T]):
    def __init__(self, get_distance: Callable[[T, T], float]) -> None:
        self._get_distance = get_distance

    def generate_clusters(self, data_objects: Iterable[T]) -> DiGraph[Cluster[T]]:
        tree: DiGraph[Cluster[T]] = DiGraph()
        clusters: List[Cluster[T]] = []
        for data_object in data_objects:
            cluster = Cluster[T](data_object, description=str(data_object))
            clusters.append(cluster)
            tree.add_node(cluster, cluster=cluster)

        distances: Dict[FrozenSet[Cluster[T]], float] = {}
        heights: Dict[Cluster[T], float] = {}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = self._get_distance(
                    next(iter(clusters[i].data_objects)), next(iter(clusters[j].data_objects))
                )
                distances[frozenset([clusters[i], clusters[j]])] = distance
            heights[clusters[i]] = 0

        while len(clusters) >= 2:
            min_i = 0
            min_j = 0
            min_dist = sys.float_info.max
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = distances[frozenset([clusters[i], clusters[j]])]
                    if dist < min_dist:
                        min_dist = dist
                        min_i = i
                        min_j = j

            i_cluster = clusters[min_i]
            j_cluster = clusters[min_j]
            del distances[frozenset([i_cluster, j_cluster])]

            u_cluster = Cluster[T](description="BRANCH")
            tree.add_node(u_cluster, cluster=u_cluster)

            height = min_dist / 2
            heights[u_cluster] = height

            i_count = get_all_data_objects_count(tree, i_cluster)
            i_len = height - heights[i_cluster]
            if i_len <= 0 and tree.out_degree(i_cluster) > 0:
                for _, target, data in tree.out_edges(i_cluster, data=True):
                    tree.add_edge(u_cluster, target, weight=data["weight"])
                tree.remove_node(i_cluster)
            else:
                tree.remove_edges_from(tree.in_edges(i_cluster))
                tree.add_edge(u_cluster, i_cluster, weight=max(i_len, 0))
            j_count = get_all_data_objects_count(tree, j_cluster)
            j_len = height - heights[j_cluster]
            if j_len <= 0 and tree.out_degree(j_cluster) > 0:
                for _, target, data in tree.out_edges(j_cluster, data=True):
                    tree.add_edge(u_cluster, target, weight=data["weight"])
                tree.remove_node(j_cluster)
            else:
                tree.remove_edges_from(tree.in_edges(j_cluster))
                tree.add_edge(u_cluster, j_cluster, weight=max(j_len, 0))

            i_weight = i_count / (i_count + j_count)
            j_weight = j_count / (i_count + j_count)
            for k_cluster in clusters:
                if k_cluster is i_cluster or k_cluster is j_cluster:
                    continue
                ki_key = frozenset([k_cluster, i_cluster])
                kj_key = frozenset([k_cluster, j_cluster])
                distances[frozenset([k_cluster, u_cluster])] = (i_weight * distances[ki_key]) + (
                    j_weight * distances[kj_key]
                )
                del distances[ki_key]
                del distances[kj_key]
            del clusters[min_j]
            del clusters[min_i]
            clusters.append(u_cluster)

        return tree


def get_all_data_objects_count(tree: DiGraph[Cluster[T]], cluster: Cluster[T]) -> int:
    if tree.out_degree(cluster) == 0:
        return len(cluster.data_objects)
    return sum(
        (get_all_data_objects_count(tree, edge[1]) for edge in tree.out_edges(cluster)), len(cluster.data_objects)
    )

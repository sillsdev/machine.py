from __future__ import annotations

import sys
from typing import Callable, Dict, FrozenSet, Iterable, List, TypeVar

from networkx import DiGraph, Graph

from .cluster import Cluster
from .unrooted_hierarchical_clusterer import UnrootedHierarchicalClusterer

T = TypeVar("T")


class NeighborJoiningClusterer(UnrootedHierarchicalClusterer[T]):
    def __init__(self, get_distance: Callable[[T, T], float]) -> None:
        self._get_distance = get_distance

    def generate_clusters(self, data_objects: Iterable[T]) -> Graph[Cluster[T]]:
        tree: DiGraph[Cluster[T]] = DiGraph()
        clusters: List[Cluster[T]] = []
        for data_object in data_objects:
            cluster = Cluster[T](data_object, description=str(data_object))
            clusters.append(cluster)
            tree.add_node(cluster, cluster=cluster)

        distances: Dict[FrozenSet[Cluster[T]], float] = {}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = self._get_distance(
                    next(iter(clusters[i].data_objects)), next(iter(clusters[j].data_objects))
                )
                distances[frozenset([clusters[i], clusters[j]])] = distance

        while len(clusters) > 2:
            r = {
                c: sum(distances[frozenset([c, oc])] / (len(clusters) - 2) for oc in clusters if oc is not c)
                for c in clusters
            }
            min_i = 0
            min_j = 0
            min_dist = 0
            min_q = sys.float_info.max
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = distances[frozenset([clusters[i], clusters[j]])]
                    q = dist - r[clusters[i]] - r[clusters[j]]
                    if q < min_q:
                        min_q = q
                        min_dist = dist
                        min_i = i
                        min_j = j

            i_cluster = clusters[min_i]
            j_cluster = clusters[min_j]
            del distances[frozenset([i_cluster, j_cluster])]

            u_cluster = Cluster[T](description="BRANCH")
            tree.add_node(u_cluster, cluster=u_cluster)

            i_len = (min_dist / 2) + ((r[i_cluster] - r[j_cluster]) / 2)
            if i_len <= 0 and tree.out_degree(i_cluster) > 0:
                for _, target, data in tree.out_edges(i_cluster, data=True):
                    tree.add_edge(u_cluster, target, weight=data["weight"])
                tree.remove_node(i_cluster)
            else:
                tree.remove_edges_from(tree.in_edges(i_cluster))
                tree.add_edge(u_cluster, i_cluster, weight=max(i_len, 0))
            j_len = min_dist - i_len
            if j_len <= 0 and tree.out_degree(j_cluster) > 0:
                for _, target, data in tree.out_edges(j_cluster, data=True):
                    tree.add_edge(u_cluster, target, weight=data["weight"])
                tree.remove_node(j_cluster)
            else:
                tree.remove_edges_from(tree.in_edges(j_cluster))
                tree.add_edge(u_cluster, j_cluster, weight=max(j_len, 0))

            for k_cluster in clusters:
                if k_cluster is i_cluster or k_cluster is j_cluster:
                    continue
                ki_key = frozenset([k_cluster, i_cluster])
                kj_key = frozenset([k_cluster, j_cluster])
                distances[frozenset([k_cluster, u_cluster])] = (distances[ki_key] + distances[kj_key] - min_dist) / 2
                del distances[ki_key]
                del distances[kj_key]
            del clusters[min_j]
            del clusters[min_i]
            clusters.append(u_cluster)

        if len(clusters) == 2:
            tree.add_edge(clusters[1], clusters[0], weight=distances[frozenset([clusters[0], clusters[1]])])
            del clusters[0]

        return tree.to_undirected()

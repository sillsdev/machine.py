from __future__ import annotations

import numpy as np
from networkx import Graph, is_isomorphic
from networkx.algorithms.isomorphism import numerical_edge_match

from machine.clusterers import Cluster, NeighborJoiningClusterer


def test_cluster() -> None:
    matrix = np.array([[0, 1, 2, 3, 3], [1, 0, 2, 3, 3], [2, 2, 0, 3, 3], [3, 3, 3, 0, 1], [3, 3, 3, 1, 0]])
    nj = NeighborJoiningClusterer[str](lambda o1, o2: float(matrix[ord(o1) - ord("A")][ord(o2) - ord("A")]))
    tree = nj.generate_clusters(["A", "B", "C", "D", "E"])

    vertices = {
        "root": Cluster[str](description="root"),
        "A": Cluster[str]("A", description="A"),
        "B": Cluster[str]("B", description="B"),
        "C": Cluster[str]("C", description="C"),
        "D": Cluster[str]("D", description="D"),
        "E": Cluster[str]("E", description="E"),
        "DE": Cluster[str](description="DE"),
        "AB": Cluster[str](description="AB"),
    }
    expected_tree: Graph[Cluster[str]] = Graph()
    expected_tree.add_nodes_from([(v, {"cluster": v}) for v in vertices.values()])
    expected_tree.add_weighted_edges_from(
        [
            (vertices["root"], vertices["C"], 1.0),
            (vertices["root"], vertices["DE"], 1.5),
            (vertices["root"], vertices["AB"], 0.5),
            (vertices["DE"], vertices["D"], 0.5),
            (vertices["DE"], vertices["E"], 0.5),
            (vertices["AB"], vertices["A"], 0.5),
            (vertices["AB"], vertices["B"], 0.5),
        ]
    )

    assert is_isomorphic(
        tree, expected_tree, node_match=cluster_node_match, edge_match=numerical_edge_match("weight", 0)
    )


def test_cluster_no_data_objects() -> None:
    nj = NeighborJoiningClusterer[str](lambda o1, o2: 0)
    tree = nj.generate_clusters([])
    assert tree.number_of_edges() == 0


def test_cluster_one_data_object() -> None:
    nj = NeighborJoiningClusterer[str](lambda o1, o2: 0)
    tree = nj.generate_clusters(["A"])
    assert tree.number_of_nodes() == 1
    assert tree.number_of_edges() == 0


def test_cluster_two_data_objects() -> None:
    nj = NeighborJoiningClusterer[str](lambda o1, o2: 1)
    tree = nj.generate_clusters(["A", "B"])

    vertices = {"A": Cluster[str]("A", description="A"), "B": Cluster[str]("B", description="B")}
    expected_tree: Graph[Cluster[str]] = Graph()
    expected_tree.add_nodes_from([(v, {"cluster": v}) for v in vertices.values()])
    expected_tree.add_weighted_edges_from([(vertices["A"], vertices["B"], 1.0)])

    assert is_isomorphic(
        tree, expected_tree, node_match=cluster_node_match, edge_match=numerical_edge_match("weight", 0)
    )


def cluster_node_match(n1: dict, n2: dict) -> bool:
    return n1["cluster"].data_objects == n2["cluster"].data_objects

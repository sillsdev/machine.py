from __future__ import annotations

import numpy as np
from networkx import DiGraph, is_isomorphic
from networkx.algorithms.isomorphism import numerical_edge_match

from machine.clusterers import Cluster, UpgmaClusterer


def test_cluster() -> None:
    matrix = np.array(
        [
            [0, 2, 4, 6, 6, 8],
            [2, 0, 4, 6, 6, 8],
            [4, 4, 0, 6, 6, 8],
            [6, 6, 6, 0, 4, 8],
            [6, 6, 6, 4, 0, 8],
            [8, 8, 8, 8, 8, 0],
        ]
    )
    upgma = UpgmaClusterer[str](lambda o1, o2: float(matrix[ord(o1) - ord("A")][ord(o2) - ord("A")]))
    tree = upgma.generate_clusters(["A", "B", "C", "D", "E", "F"])

    vertices = {
        "root": Cluster[str](description="root"),
        "A": Cluster[str]("A", description="A"),
        "B": Cluster[str]("B", description="B"),
        "C": Cluster[str]("C", description="C"),
        "D": Cluster[str]("D", description="D"),
        "E": Cluster[str]("E", description="E"),
        "F": Cluster[str]("F", description="F"),
        "ABCDE": Cluster[str](description="ABCDE"),
        "ABC": Cluster[str](description="ABC"),
        "AB": Cluster[str](description="AB"),
        "DE": Cluster[str](description="DE"),
    }
    expected_tree: DiGraph[Cluster[str]] = DiGraph()
    expected_tree.add_nodes_from([(v, {"cluster": v}) for v in vertices.values()])
    expected_tree.add_weighted_edges_from(
        [
            (vertices["root"], vertices["ABCDE"], 1),
            (vertices["root"], vertices["F"], 4),
            (vertices["ABCDE"], vertices["ABC"], 1),
            (vertices["ABCDE"], vertices["DE"], 1),
            (vertices["ABC"], vertices["AB"], 1),
            (vertices["ABC"], vertices["C"], 2),
            (vertices["AB"], vertices["A"], 1),
            (vertices["AB"], vertices["B"], 1),
            (vertices["DE"], vertices["D"], 2),
            (vertices["DE"], vertices["E"], 2),
        ]
    )

    assert is_isomorphic(
        tree, expected_tree, node_match=cluster_node_match, edge_match=numerical_edge_match("weight", 0)
    )


def test_cluster_no_data_objects() -> None:
    upgma = UpgmaClusterer[str](lambda o1, o2: 0)
    tree = upgma.generate_clusters([])
    assert tree.number_of_edges() == 0


def test_cluster_one_data_object() -> None:
    upgma = UpgmaClusterer[str](lambda o1, o2: 0)
    tree = upgma.generate_clusters(["A"])
    assert tree.number_of_nodes() == 1
    assert tree.number_of_edges() == 0


def test_cluster_two_data_objects() -> None:
    upgma = UpgmaClusterer[str](lambda o1, o2: 1)
    tree = upgma.generate_clusters(["A", "B"])

    vertices = {
        "root": Cluster[str](description="root"),
        "A": Cluster[str]("A", description="A"),
        "B": Cluster[str]("B", description="B"),
    }
    expected_tree: DiGraph[Cluster[str]] = DiGraph()
    expected_tree.add_nodes_from([(v, {"cluster": v}) for v in vertices.values()])
    expected_tree.add_weighted_edges_from(
        [(vertices["root"], vertices["A"], 0.5), (vertices["root"], vertices["B"], 0.5)]
    )

    assert is_isomorphic(
        tree, expected_tree, node_match=cluster_node_match, edge_match=numerical_edge_match("weight", 0)
    )


def cluster_node_match(n1: dict, n2: dict) -> bool:
    return n1["cluster"].data_objects == n2["cluster"].data_objects

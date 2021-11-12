import numpy as np

from machine.clusterers import FlatUpgmaClusterer


def test_cluster() -> None:
    matrix = np.array(
        [
            [0.00, 0.50, 0.67, 0.80, 0.20],
            [0.50, 0.00, 0.40, 0.70, 0.60],
            [0.67, 0.40, 0.00, 0.80, 0.80],
            [0.80, 0.70, 0.80, 0.00, 0.30],
            [0.20, 0.60, 0.80, 0.30, 0.00],
        ]
    )
    fupgma = FlatUpgmaClusterer[str](lambda o1, o2: float(matrix[ord(o1) - ord("A")][ord(o2) - ord("A")]), 0.5)
    clusters = fupgma.generate_clusters(["A", "B", "C", "D", "E"])

    expected = set([frozenset(["B", "C"]), frozenset(["A", "E", "D"])])

    assert set(c.data_objects for c in clusters) == expected

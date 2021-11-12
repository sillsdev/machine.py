from typing import Callable, Optional

from ...classes import Graph

def could_be_isomorphic(G1: Graph, G2: Graph): ...

graph_could_be_isomorphic = could_be_isomorphic

def fast_could_be_isomorphic(G1: Graph, G2: Graph): ...

fast_graph_could_be_isomorphic = fast_could_be_isomorphic

def faster_could_be_isomorphic(G1: Graph, G2: Graph): ...

faster_graph_could_be_isomorphic = faster_could_be_isomorphic

def is_isomorphic(
    G1: Graph,
    G2: Graph,
    node_match: Optional[Callable[[dict, dict], bool]] = ...,
    edge_match: Optional[Callable[[dict, dict], bool]] = ...,
) -> bool: ...

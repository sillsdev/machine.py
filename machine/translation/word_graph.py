from dataclasses import dataclass, field
from typing import AbstractSet, Dict, Iterable, List, Sequence

from .word_graph_arc import WordGraphArc

EMPTY_ARC_INDICES: Sequence[int] = []


@dataclass(frozen=True)
class StateInfo:
    prev_arc_indices: List[int] = field(default_factory=list)
    next_arc_indices: List[int] = field(default_factory=list)


class WordGraph:
    def __init__(
        self, arcs: Iterable[WordGraphArc] = [], final_states: Iterable[int] = [], initial_state_score: float = 0
    ) -> None:
        self._states: Dict[int, StateInfo] = {}
        arc_list: List[WordGraphArc] = []
        max_state = -1
        for arc in arcs:
            if arc.next_state > max_state:
                max_state = arc.next_state
            if arc.prev_state > max_state:
                max_state = arc.prev_state

            arc_index = len(arc_list)
            self._get_or_create_state_info(arc.prev_state).next_arc_indices.append(arc_index)
            self._get_or_create_state_info(arc.next_state).prev_arc_indices.append(arc_index)
            arc_list.append(arc)
        self._arcs = arc_list
        self._state_count = max_state + 1
        self._final_states = set(final_states)
        self._initial_state_score = initial_state_score

    @property
    def initial_state_score(self) -> float:
        return self._initial_state_score

    @property
    def arcs(self) -> Sequence[WordGraphArc]:
        return self._arcs

    @property
    def state_count(self) -> int:
        return self._state_count

    @property
    def final_states(self) -> AbstractSet[int]:
        return self._final_states

    @property
    def is_empty(self) -> bool:
        return len(self._arcs) == 0

    def get_prev_arc_indices(self, state: int) -> Sequence[int]:
        state_info = self._states.get(state)
        if state_info is None:
            return EMPTY_ARC_INDICES
        return state_info.prev_arc_indices

    def get_next_arc_indices(self, state: int) -> Sequence[int]:
        state_info = self._states.get(state)
        if state_info is None:
            return EMPTY_ARC_INDICES
        return state_info.next_arc_indices

    def _get_or_create_state_info(self, state: int) -> StateInfo:
        state_info = self._states.get(state)
        if state_info is None:
            state_info = StateInfo()
            self._states[state] = state_info
        return state_info

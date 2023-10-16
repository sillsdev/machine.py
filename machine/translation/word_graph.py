from dataclasses import dataclass, field
from typing import AbstractSet, Dict, Generator, Iterable, List, Sequence, Set, Tuple

from ..statistics.log_space import LOG_SPACE_ZERO, log_space_multiple
from .word_graph_arc import WordGraphArc

EMPTY_ARC_INDICES: Sequence[int] = []
WORD_GRAPH_INITIAL_STATE = 0


@dataclass(frozen=True)
class StateInfo:
    prev_arc_indices: List[int] = field(default_factory=list)
    next_arc_indices: List[int] = field(default_factory=list)


class WordGraph:
    def __init__(
        self,
        source_tokens: Iterable[str],
        arcs: Iterable[WordGraphArc] = [],
        final_states: Iterable[int] = [],
        initial_state_score: float = 0,
    ) -> None:
        self._source_tokens = list(source_tokens)
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
    def source_tokens(self) -> Sequence[str]:
        return self._source_tokens

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

    def compute_rest_scores(self) -> List[float]:
        rest_scores: List[float] = [LOG_SPACE_ZERO] * self._state_count
        for state in self._final_states:
            rest_scores[state] = self._initial_state_score

        for arc in reversed(self._arcs):
            score = log_space_multiple(arc.score, rest_scores[arc.next_state])
            if score > rest_scores[arc.prev_state]:
                rest_scores[arc.prev_state] = score
        return rest_scores

    def get_best_path_from_state_to_final_state(self, state: int) -> Iterable[WordGraphArc]:
        arcs = list(self._get_best_path_from_final_state_to_state(state))
        arcs.reverse()
        return arcs

    def _get_best_path_from_final_state_to_state(self, state: int) -> Generator[WordGraphArc, None, None]:
        prev_scores, state_best_pred_arcs = self._compute_prev_scores(state)

        best_final_state_score: float = LOG_SPACE_ZERO
        best_final_state = WORD_GRAPH_INITIAL_STATE
        for final_state in self._final_states:
            score = prev_scores[final_state]
            if best_final_state_score < score:
                best_final_state = final_state
                best_final_state_score = score

        if best_final_state in self._final_states:
            cur_state = best_final_state
            end = False
            while not end:
                if cur_state == state:
                    end = True
                else:
                    arc_index = state_best_pred_arcs[cur_state]
                    arc = self.arcs[arc_index]
                    yield arc
                    cur_state = arc.prev_state

    def _compute_prev_scores(self, state: int) -> Tuple[List[float], List[int]]:
        if self.is_empty:
            return [], []

        prev_scores: List[float] = [LOG_SPACE_ZERO] * self._state_count
        state_best_prev_arcs = [0] * self._state_count

        if state == WORD_GRAPH_INITIAL_STATE:
            prev_scores[WORD_GRAPH_INITIAL_STATE] = self.initial_state_score
        else:
            prev_scores[state] = 0

        accessible_states: Set[int] = {state}
        for arc_index in range(len(self.arcs)):
            arc = self.arcs[arc_index]
            if arc.prev_state in accessible_states:
                score = log_space_multiple(arc.score, prev_scores[arc.prev_state])
                if score > prev_scores[arc.next_state]:
                    prev_scores[arc.next_state] = score
                    state_best_prev_arcs[arc.next_state] = arc_index
                accessible_states.add(arc.next_state)
            else:
                if arc.next_state not in accessible_states:
                    prev_scores[arc.next_state] = LOG_SPACE_ZERO
        return prev_scores, state_best_prev_arcs

    def _get_or_create_state_info(self, state: int) -> StateInfo:
        state_info = self._states.get(state)
        if state_info is None:
            state_info = StateInfo()
            self._states[state] = state_info
        return state_info

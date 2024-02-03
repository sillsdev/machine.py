from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import total_ordering
from queue import PriorityQueue
from typing import Any, Generator, List, Sequence, Set

from ..statistics.log_space import LOG_SPACE_ZERO
from ..tokenization.detokenizer import Detokenizer
from .ecm_score_info import EcmScoreInfo
from .error_correction_model import ErrorCorrectionModel
from .translation_result import TranslationResult
from .translation_result_builder import TranslationResultBuilder
from .word_alignment_matrix import WordAlignmentMatrix
from .word_graph import WORD_GRAPH_INITIAL_STATE, WordGraph
from .word_graph_arc import WordGraphArc


class ErrorCorrectionWordGraphProcessor:
    def __init__(
        self,
        ecm: ErrorCorrectionModel,
        target_detokenizer: Detokenizer[str, str],
        word_graph: WordGraph,
        ecm_weight: float = 1,
        word_graph_weight: float = 1,
    ) -> None:
        self.confidence_threshold = 0.0
        self._ecm = ecm
        self._target_detokenizer = target_detokenizer
        self._word_graph = word_graph
        self._ecm_weight = ecm_weight
        self._word_graph_weight = word_graph_weight

        self._rest_scores = self._word_graph.compute_rest_scores()
        self._state_ecm_score_infos: List[EcmScoreInfo] = []
        self._arc_ecm_score_infos: List[List[EcmScoreInfo]] = []
        self._state_best_scores: List[List[float]] = []
        self._state_word_graph_scores: List[float] = []
        self._state_best_prev_arcs: List[List[int]] = []
        self._states_involved_in_arcs: Set[int] = set()
        self._prev_prefix: List[str] = []
        self._prev_is_last_word_complete = False

        self._init_states()
        self._init_arcs()

    @property
    def ecm_weight(self) -> float:
        return self._ecm_weight

    @property
    def word_graph_weight(self) -> float:
        return self._word_graph_weight

    def _init_states(self) -> None:
        for _ in range(self._word_graph.state_count):
            self._state_ecm_score_infos.append(EcmScoreInfo())
            self._state_word_graph_scores.append(LOG_SPACE_ZERO)
            self._state_best_scores.append([])
            self._state_best_prev_arcs.append([])

        if not self._word_graph.is_empty:
            self._ecm.setup_initial_esi(self._state_ecm_score_infos[WORD_GRAPH_INITIAL_STATE])
            self._update_initial_state_best_scores()

    def _init_arcs(self) -> None:
        for arc_index in range(len(self._word_graph.arcs)):
            arc = self._word_graph.arcs[arc_index]

            # init ecm score info for each word of arc
            prev_esi = self._state_ecm_score_infos[arc.prev_state]
            esis: List[EcmScoreInfo] = []
            for word in arc.target_tokens:
                esi = EcmScoreInfo()
                self._ecm.setup_esi(esi, prev_esi, word)
                esis.append(esi)
                prev_esi = esi
            self._arc_ecm_score_infos.append(esis)

            # init best scores for the arc's successive state
            self._update_state_best_scores(arc_index, 0)

            self._states_involved_in_arcs.add(arc.prev_state)
            self._states_involved_in_arcs.add(arc.next_state)

    def _update_initial_state_best_scores(self) -> None:
        esi = self._state_ecm_score_infos[WORD_GRAPH_INITIAL_STATE]

        self._state_word_graph_scores[WORD_GRAPH_INITIAL_STATE] = self._word_graph.initial_state_score

        best_scores = self._state_best_scores[WORD_GRAPH_INITIAL_STATE]
        best_prev_arcs = self._state_best_prev_arcs[WORD_GRAPH_INITIAL_STATE]

        best_scores.clear()
        best_prev_arcs.clear()

        for score in esi.scores:
            best_scores.append(
                (self.ecm_weight * -score) + (self.word_graph_weight * self._word_graph.initial_state_score)
            )
            best_prev_arcs.append(sys.maxsize)

    def _update_state_best_scores(self, arc_index: int, prefix_diff_size: int) -> None:
        arc = self._word_graph.arcs[arc_index]
        arc_esis = self._arc_ecm_score_infos[arc_index]

        prev_esi = self._state_ecm_score_infos[arc.prev_state] if len(arc_esis) == 0 else arc_esis[-1]

        word_graph_score = self._state_word_graph_scores[arc.prev_state] + arc.score

        next_state_best_scores = self._state_best_scores[arc.next_state]
        next_state_best_prev_arcs = self._state_best_prev_arcs[arc.next_state]

        positions: List[int] = []
        start_pos = 0 if prefix_diff_size == 0 else len(prev_esi.scores) - prefix_diff_size
        for i in range(start_pos, len(prev_esi.scores)):
            new_score = (self.ecm_weight * -prev_esi.scores[i]) + (self.word_graph_weight * word_graph_score)

            if i == len(next_state_best_scores) or next_state_best_scores[i] < new_score:
                _add_or_replace(next_state_best_scores, i, new_score)
                positions.append(i)
                _add_or_replace(next_state_best_prev_arcs, i, arc_index)

        self._state_ecm_score_infos[arc.next_state].update_positions(prev_esi, positions)

        if word_graph_score > self._state_word_graph_scores[arc.next_state]:
            self._state_word_graph_scores[arc.next_state] = word_graph_score

    def correct(self, prefix: Sequence[str], is_last_word_complete: bool) -> None:
        # get valid portion of the processed prefix vector
        valid_proc_prefix_count = 0
        for i in range(len(self._prev_prefix)):
            if i >= len(prefix):
                break

            if i == len(self._prev_prefix) - 1 and i == len(prefix) - 1:
                if self._prev_prefix[i] == prefix[i] and self._prev_is_last_word_complete == is_last_word_complete:
                    valid_proc_prefix_count += 1
            elif self._prev_prefix[i] == prefix[i]:
                valid_proc_prefix_count += 1

        diff_size = len(self._prev_prefix) - valid_proc_prefix_count
        if diff_size > 0:
            # adjust size of info for arcs
            for esis in self._arc_ecm_score_infos:
                for esi in esis:
                    for i in range(diff_size):
                        esi.remove_last()

            # adjust size of info for states
            for state in self._states_involved_in_arcs:
                for i in range(diff_size):
                    self._state_ecm_score_infos[state].remove_last()
                    self._state_best_scores[state].pop()
                    self._state_best_prev_arcs[state].pop()

        # get difference between prefix and valid portion of processed prefix
        prefix_diff: List[str] = []
        for i in range(len(prefix) - valid_proc_prefix_count):
            prefix_diff.append(prefix[valid_proc_prefix_count + i])

        # process word-graph given prefix difference
        self._process_word_graph_prefix_diff(prefix_diff, is_last_word_complete)

        self._prev_prefix = list(prefix)
        self._prev_is_last_word_complete = is_last_word_complete

    def get_results(self) -> Generator[TranslationResult, None, None]:
        queue = self._get_hypotheses()

        for hypothesis in self._search(queue):
            builder = TranslationResultBuilder(self._word_graph.source_tokens, self._target_detokenizer)
            self._build_correction_from_hypothesis(
                builder, self._prev_prefix, self._prev_is_last_word_complete, hypothesis
            )
            yield builder.to_result()

    def _search(self, queue: PriorityQueue) -> Generator[_Hypothesis, None, None]:
        while queue.not_empty:
            hypothesis: _Hypothesis = queue.get()
            last_state = hypothesis.start_state if len(hypothesis.arcs) == 0 else hypothesis.arcs[-1].next_state

            if last_state in self._word_graph.final_states:
                yield hypothesis
            elif self.confidence_threshold <= 0:
                hypothesis.arcs.extend(self._word_graph.get_best_path_from_state_to_final_state(last_state))
                yield hypothesis
            else:
                score = hypothesis.score - (self.word_graph_weight * self._rest_scores[last_state])
                arc_indices = self._word_graph.get_next_arc_indices(last_state)
                enqueued_arc = False
                for i in range(len(arc_indices)):
                    arc_index = arc_indices[i]
                    arc = self._word_graph.arcs[arc_index]
                    if self._is_arc_pruned(arc):
                        continue

                    new_hypothesis = hypothesis
                    if i < len(arc_indices) - 1:
                        new_hypothesis = new_hypothesis.clone()
                    new_hypothesis.score = score
                    new_hypothesis.score += arc.score
                    new_hypothesis.score += self._rest_scores[arc.next_state]
                    new_hypothesis.arcs.append(arc)
                    queue.put(new_hypothesis)
                    enqueued_arc = True

                if not enqueued_arc and (hypothesis.start_arc_index != -1 or len(hypothesis.arcs) > 0):
                    hypothesis.arcs.extend(self._word_graph.get_best_path_from_state_to_final_state(last_state))
                    yield hypothesis

    def _get_hypotheses(self) -> PriorityQueue:
        queue = PriorityQueue(maxsize=1000)

        # add hypotheses starting before each word in each arc
        for arc_index in range(len(self._word_graph.arcs)):
            arc = self._word_graph.arcs[arc_index]
            if not self._is_arc_pruned(arc):
                word_graph_score = self._state_word_graph_scores[arc.prev_state] + arc.score

                for i in range(-1, len(arc.target_tokens) - 1):
                    esi = (
                        self._state_ecm_score_infos[arc.prev_state]
                        if i == -1
                        else self._arc_ecm_score_infos[arc_index][i]
                    )
                    score = (
                        (self.word_graph_weight * word_graph_score)
                        + (self.ecm_weight * -esi.scores[-1])
                        + (self.word_graph_weight * self._rest_scores[arc.next_state])
                    )
                    queue.put(_Hypothesis(score, arc.next_state, arc_index, i))

        # add hypotheses starting before each final state
        for state in self._word_graph.final_states:
            rest_score = self._rest_scores[state]
            best_scores = self._state_best_scores[state]

            score = best_scores[-1] + (self.word_graph_weight * rest_score)
            queue.put(_Hypothesis(score, state))

        return queue

    def _is_arc_pruned(self, arc: WordGraphArc) -> bool:
        return not arc.is_unknown and any(c < self.confidence_threshold for c in arc.confidences)

    def _build_correction_from_hypothesis(
        self,
        builder: TranslationResultBuilder,
        prefix: Sequence[str],
        is_last_word_complete: bool,
        hypothesis: _Hypothesis,
    ) -> None:
        if hypothesis.start_arc_index == -1:
            self._add_best_uncorrected_prefix_state(builder, len(prefix), hypothesis.start_state)
            uncorrected_prefix_len = len(builder.target_tokens)
        else:
            self._add_best_uncorrected_prefix_sub_state(
                builder, len(prefix), hypothesis.start_arc_index, hypothesis.start_arc_word_index
            )
            first_arc = self._word_graph.arcs[hypothesis.start_arc_index]
            uncorrected_prefix_len = (
                len(builder.target_tokens) - (len(first_arc.target_tokens) - hypothesis.start_arc_word_index) + 1
            )

        alignment_cols_to_add_count = self._ecm.correct_prefix(
            builder, uncorrected_prefix_len, prefix, is_last_word_complete
        )

        for arc in hypothesis.arcs:
            self._update_correction_from_arc(builder, arc, alignment_cols_to_add_count)
            alignment_cols_to_add_count = 0

    def _add_best_uncorrected_prefix_state(
        self, builder: TranslationResultBuilder, proc_prefix_pos: int, state: int
    ) -> None:
        arcs: List[WordGraphArc] = []

        cur_state = state
        cur_proc_prefix_pos = proc_prefix_pos
        while cur_state != WORD_GRAPH_INITIAL_STATE:
            arc_index = self._state_best_prev_arcs[cur_state][cur_proc_prefix_pos]
            arc = self._word_graph.arcs[arc_index]

            for i in range(len(arc.target_tokens) - 1, -1, -1):
                pred_prefix_words = self._arc_ecm_score_infos[arc_index][i].get_last_ins_prefix_word_from_esi()
                cur_proc_prefix_pos = pred_prefix_words[cur_proc_prefix_pos]

            arcs.append(arc)

            cur_state = arc.prev_state

        for arc in reversed(arcs):
            self._update_correction_from_arc(builder, arc, 0)

    def _add_best_uncorrected_prefix_sub_state(
        self, builder: TranslationResultBuilder, proc_prefix_pos: int, arc_index: int, arc_word_index: int
    ) -> None:
        arc = self._word_graph.arcs[arc_index]

        cur_proc_prefix_pos = proc_prefix_pos
        for i in range(arc_word_index, -1, -1):
            pred_prefix_words = self._arc_ecm_score_infos[arc_index][i].get_last_ins_prefix_word_from_esi()
            cur_proc_prefix_pos = pred_prefix_words[cur_proc_prefix_pos]

        self._add_best_uncorrected_prefix_state(builder, cur_proc_prefix_pos, arc.prev_state)

        self._update_correction_from_arc(builder, arc, 0)

    def _update_correction_from_arc(
        self, builder: TranslationResultBuilder, arc: WordGraphArc, alignment_cols_to_add_count: int
    ) -> None:
        for i in range(len(arc.target_tokens)):
            builder.append_token(arc.target_tokens[i], arc.sources[i], arc.confidences[i])

        alignment = arc.alignment
        if alignment_cols_to_add_count > 0:
            new_alignment = WordAlignmentMatrix.from_word_pairs(
                alignment.row_count, alignment.column_count + alignment_cols_to_add_count
            )
            for j in range(alignment.column_count):
                for i in range(alignment.row_count):
                    new_alignment[i, alignment_cols_to_add_count + j] = alignment[i, j]
            alignment = new_alignment

        builder.mark_phrase(arc.source_segment_range, alignment)

    def _process_word_graph_prefix_diff(self, prefix_diff: List[str], is_last_word_complete: bool) -> None:
        if len(prefix_diff) == 0:
            return

        if not self._word_graph.is_empty:
            prev_initial_esi = self._state_ecm_score_infos[WORD_GRAPH_INITIAL_STATE]
            self._ecm.extend_initial_esi(
                self._state_ecm_score_infos[WORD_GRAPH_INITIAL_STATE], prev_initial_esi, prefix_diff
            )
            self._update_initial_state_best_scores()

        for arc_index in range(len(self._word_graph.arcs)):
            arc = self._word_graph.arcs[arc_index]

            # extend ecm score info for each word of arc
            prev_esi = self._state_ecm_score_infos[arc.prev_state]
            esis = self._arc_ecm_score_infos[arc_index]
            while len(esis) < len(arc.target_tokens):
                esis.append(EcmScoreInfo())
            for i in range(len(arc.target_tokens)):
                esi = esis[i]
                self._ecm.extend_esi(
                    esi,
                    prev_esi,
                    "" if arc.is_unknown else arc.target_tokens[i],
                    prefix_diff,
                    is_last_word_complete,
                )
                prev_esi = esi

            # update best scores for the arc's successive state
            self._update_state_best_scores(arc_index, len(prefix_diff))


@dataclass
@total_ordering
class _Hypothesis:
    score: float
    start_state: int
    start_arc_index: int = -1
    start_arc_word_index: int = -1
    arcs: List[WordGraphArc] = field(default_factory=list)

    def __lt__(self, other: _Hypothesis) -> bool:
        return self.score > other.score

    def __le__(self, other: _Hypothesis) -> bool:
        return self.score >= other.score

    def __gt__(self, other: _Hypothesis) -> bool:
        return self.score < other.score

    def __ge__(self, other: _Hypothesis) -> bool:
        return self.score <= other.score

    def clone(self) -> _Hypothesis:
        return _Hypothesis(
            self.score, self.start_state, self.start_arc_index, self.start_arc_word_index, list(self.arcs)
        )


def _add_or_replace(list: list, index: int, item: Any) -> None:
    if index > len(list):
        raise ValueError("index out of range")

    if index == len(list):
        list.append(item)
    else:
        list[index] = item

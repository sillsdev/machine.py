from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import thot.translation as tt

from ...annotations.range import Range
from ...corpora import ParallelTextCorpus
from ...utils.typeshed import StrPath
from ..fuzzy_edit_distance_word_alignment_method import FuzzyEditDistanceWordAlignmentMethod
from ..ibm1_word_confidence_estimator import Ibm1WordConfidenceEstimator
from ..interactive_translation_model import InteractiveTranslationModel
from ..translation_result import TranslationResult
from ..translation_result_builder import TranslationResultBuilder
from ..translation_sources import TranslationSources
from ..word_aligner import WordAligner
from ..word_alignment_matrix import WordAlignmentMatrix
from ..word_alignment_method import WordAlignmentMethod
from ..word_graph import WordGraph
from ..word_graph_arc import WordGraphArc
from .thot_smt_model_trainer import ThotSmtModelTrainer
from .thot_smt_parameters import ThotSmtParameters
from .thot_symmetrized_word_alignment_model import ThotSymmetrizedWordAlignmentModel
from .thot_utils import escape_tokens, load_smt_decoder, load_smt_model, unescape_token, unescape_tokens
from .thot_word_alignment_model import ThotWordAlignmentModel
from .thot_word_alignment_model_type import ThotWordAlignmentModelType
from .thot_word_alignment_model_utils import create_thot_word_alignment_model


class _Trainer(ThotSmtModelTrainer):
    def __init__(
        self,
        smt_model: ThotSmtModel,
        corpus: ParallelTextCorpus,
        config: Union[ThotSmtParameters, StrPath],
    ) -> None:
        super().__init__(smt_model.word_alignment_model_type, corpus, config)
        self._smt_model = smt_model

    def save(self) -> None:
        self._smt_model.close()

        super().save()

        self._smt_model._parameters = self._parameters
        self._smt_model._load()


class ThotSmtModel(InteractiveTranslationModel):
    def __init__(
        self, word_alignment_model_type: ThotWordAlignmentModelType, config: Union[ThotSmtParameters, StrPath]
    ) -> None:
        if isinstance(config, ThotSmtParameters):
            self._config_filename = None
            parameters = config
        else:
            self._config_filename = Path(config)
            parameters = ThotSmtParameters.load(config)
        self._parameters = parameters

        self._word_alignment_model_type = word_alignment_model_type
        self._direct_word_alignment_model = create_thot_word_alignment_model(self._word_alignment_model_type)
        self._inverse_word_alignment_model = create_thot_word_alignment_model(self._word_alignment_model_type)
        self._symmetrized_word_alignment_model = ThotSymmetrizedWordAlignmentModel(
            self._direct_word_alignment_model, self._inverse_word_alignment_model
        )
        self._confidence_estimator = Ibm1WordConfidenceEstimator(
            self._symmetrized_word_alignment_model.get_translation_score
        )

        self._load()

        self._decoder: Optional[tt.SmtDecoder] = None
        self.word_aligner = FuzzyEditDistanceWordAlignmentMethod()

    @property
    def config_filename(self) -> Optional[Path]:
        return self._config_filename

    @property
    def parameters(self) -> ThotSmtParameters:
        return self._parameters

    @property
    def direct_word_alignment_model(self) -> ThotWordAlignmentModel:
        return self._direct_word_alignment_model

    @property
    def inverse_word_alignment_model(self) -> ThotWordAlignmentModel:
        return self._inverse_word_alignment_model

    @property
    def symmetrized_word_alignment_model(self) -> ThotSymmetrizedWordAlignmentModel:
        return self._symmetrized_word_alignment_model

    @property
    def word_alignment_model_type(self) -> ThotWordAlignmentModelType:
        return self._word_alignment_model_type

    @property
    def word_aligner(self) -> WordAligner:
        return self._word_aligner

    @word_aligner.setter
    def word_aligner(self, value: WordAligner) -> None:
        self._word_aligner = value
        if isinstance(self._word_aligner, WordAlignmentMethod):
            self._word_aligner.score_selector = self._get_word_alignment_score

    def get_word_graph(self, segment: Sequence[str]) -> WordGraph:
        decoder = self._get_decoder()
        word_graph = decoder.get_word_graph(" ".join(escape_tokens(segment)))
        return self._convert_word_graph(segment, word_graph)

    def translate(self, segment: Sequence[str]) -> TranslationResult:
        decoder = self._get_decoder()
        translation = decoder.translate(" ".join(escape_tokens(segment)))
        return self._create_result(segment, translation)

    def translate_n(self, n: int, segment: Sequence[str]) -> Sequence[TranslationResult]:
        decoder = self._get_decoder()
        n_translations = decoder.translate_n(" ".join(escape_tokens(segment)), n)
        return [self._create_result(segment, t) for t in n_translations]

    def translate_batch(self, segments: Sequence[Sequence[str]]) -> Sequence[TranslationResult]:
        decoder = self._get_decoder()
        translations = decoder.translate_batch([" ".join(escape_tokens(s)) for s in segments])
        return [self._create_result(s, t) for s, t in zip(segments, translations)]

    def translate_n_batch(self, n: int, segments: Sequence[Sequence[str]]) -> Sequence[Sequence[TranslationResult]]:
        decoder = self._get_decoder()
        all_translations = decoder.translate_n_batch([" ".join(escape_tokens(s)) for s in segments], n)
        return [
            [self._create_result(s, t) for t in n_translations] for s, n_translations in zip(segments, all_translations)
        ]

    def train_segment(
        self, source_segment: Sequence[str], target_segment: Sequence[str], sentence_start: bool = True
    ) -> None:
        decoder = self._get_decoder()
        decoder.train_sentence_pair(" ".join(escape_tokens(source_segment)), " ".join(escape_tokens(target_segment)))

    def create_trainer(self, corpus: ParallelTextCorpus) -> ThotSmtModelTrainer:
        return _Trainer(self, corpus, self._parameters if self._config_filename is None else self._config_filename)

    def save(self) -> None:
        assert self._model is not None
        self._model.print_translation_model(self._parameters.translation_model_filename_prefix)
        self._model.print_language_model(self._parameters.language_model_filename_prefix)

    def close(self) -> None:
        if self._decoder is not None:
            self._decoder.clear()
        self._decoder = None
        if self._model is not None:
            self._model.clear()
        self._model = None

    def __enter__(self) -> ThotSmtModel:
        return self

    def _get_decoder(self) -> tt.SmtDecoder:
        if self._decoder is None:
            assert self._model is not None
            self._decoder = load_smt_decoder(self._model, self._parameters)
        return self._decoder

    def _get_word_alignment_score(
        self,
        source_segment: Sequence[str],
        source_index: int,
        target_segment: Sequence[str],
        target_index: int,
    ) -> float:
        return self._symmetrized_word_alignment_model.get_translation_score(
            None if source_index == -1 else source_segment[source_index],
            None if target_index == -1 else target_segment[target_index],
        )

    def _convert_word_graph(self, segment: Sequence[str], thot_word_graph: tt.WordGraph) -> WordGraph:
        arcs: List[WordGraphArc] = []
        for thot_arc_id in range(thot_word_graph.num_arcs):
            thot_arc = thot_word_graph.get_arc(thot_arc_id)
            src_phrase_len = thot_arc.source_end_index - thot_arc.source_start_index
            words = list(unescape_tokens(thot_arc.words))
            if src_phrase_len == 1 and len(words) == 1:
                wa_matrix = WordAlignmentMatrix.from_word_pairs(1, 1, {(0, 0)})
            else:
                wa_matrix = self._word_aligner.align(
                    segment[thot_arc.source_start_index : thot_arc.source_end_index], words
                )

            arcs.append(
                WordGraphArc(
                    thot_arc.in_state,
                    thot_arc.out_state,
                    thot_arc.score,
                    words,
                    wa_matrix,
                    Range[int].create(thot_arc.source_start_index, thot_arc.source_end_index + 1),
                    [TranslationSources.NONE if thot_arc.is_unknown else TranslationSources.SMT] * len(words),
                )
            )
        word_graph = WordGraph(arcs, thot_word_graph.final_states, thot_word_graph.initial_state_score)
        self._confidence_estimator.estimate_word_graph(segment, word_graph)
        return word_graph

    def _create_result(self, source_segment: Sequence[str], data: tt.TranslationData) -> TranslationResult:
        builder = TranslationResultBuilder()

        trg_phrase_start_index = 0
        for k in range(len(data.source_segmentation)):
            source_start_index, source_end_index = data.source_segmentation[k]
            source_start_index -= 1
            target_cut = data.target_segment_cuts[k]

            for j in range(trg_phrase_start_index, target_cut):
                target_word = unescape_token(data.target[j])
                builder.append_word(
                    target_word, TranslationSources.NONE if j in data.target_unknown_words else TranslationSources.SMT
                )

            src_phrase_len = source_end_index - source_start_index
            trg_phrase_len = target_cut - trg_phrase_start_index
            if src_phrase_len == 1 and trg_phrase_len == 1:
                wa_matrix = WordAlignmentMatrix.from_word_pairs(1, 1, {(0, 0)})
            else:
                src_phrase = [""] * src_phrase_len
                for i in range(src_phrase_len):
                    src_phrase[i] = source_segment[source_start_index + i]
                trg_phrase = [""] * trg_phrase_len
                for j in range(trg_phrase_len):
                    trg_phrase[j] = data.target[trg_phrase_start_index + j]
                wa_matrix = self._word_aligner.align(src_phrase, trg_phrase)
            builder.mark_phrase(Range[int].create(source_start_index, source_end_index), wa_matrix)
            trg_phrase_start_index += trg_phrase_len

        self._confidence_estimator.estimate_translation_result(source_segment, builder)
        return builder.to_result(len(source_segment))

    def _load(self) -> None:
        self._model = load_smt_model(self._word_alignment_model_type, self._parameters)
        self._direct_word_alignment_model._set_model(self._model.direct_word_alignment_model, owned=True)
        self._inverse_word_alignment_model._set_model(self._model.inverse_word_alignment_model, owned=True)
        self._symmetrized_word_alignment_model._reset_aligner()

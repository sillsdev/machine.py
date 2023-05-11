from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import thot.translation as tt

from ...annotations.range import Range
from ...corpora import ParallelTextCorpus
from ...corpora.token_processors import lowercase
from ...tokenization.detokenizer import Detokenizer
from ...tokenization.tokenizer import Tokenizer
from ...tokenization.whitespace_detokenizer import WHITESPACE_DETOKENIZER
from ...tokenization.whitespace_tokenizer import WHITESPACE_TOKENIZER
from ...utils.typeshed import StrPath
from ..fuzzy_edit_distance_word_alignment_method import FuzzyEditDistanceWordAlignmentMethod
from ..ibm1_word_confidence_estimator import Ibm1WordConfidenceEstimator
from ..interactive_translation_model import InteractiveTranslationModel
from ..translation_result import TranslationResult
from ..translation_result_builder import TranslationResultBuilder
from ..translation_sources import TranslationSources
from ..truecaser import Truecaser
from ..word_aligner import WordAligner
from ..word_alignment_matrix import WordAlignmentMatrix
from ..word_alignment_method import WordAlignmentMethod
from ..word_graph import WordGraph
from ..word_graph_arc import WordGraphArc
from .thot_smt_model_trainer import ThotSmtModelTrainer
from .thot_smt_parameters import ThotSmtParameters
from .thot_symmetrized_word_alignment_model import ThotSymmetrizedWordAlignmentModel
from .thot_utils import load_smt_decoder, load_smt_model, to_sentence, to_target_tokens, unescape_token
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
        self,
        word_alignment_model_type: ThotWordAlignmentModelType,
        config: Union[ThotSmtParameters, StrPath],
        source_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        target_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        target_detokenizer: Detokenizer[str, str] = WHITESPACE_DETOKENIZER,
        lowercase_source: bool = False,
        lowercase_target: bool = False,
        truecaser: Optional[Truecaser] = None,
    ) -> None:
        if isinstance(config, ThotSmtParameters):
            self._config_filename = None
            parameters = config
        else:
            self._config_filename = Path(config)
            parameters = ThotSmtParameters.load(config)
        self._parameters = parameters
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.target_detokenizer = target_detokenizer
        self.lowercase_source = lowercase_source
        self.lowercase_target = lowercase_target
        self.truecaser = truecaser

        self._word_alignment_model_type = word_alignment_model_type
        self._direct_word_alignment_model = create_thot_word_alignment_model(self._word_alignment_model_type)
        self._inverse_word_alignment_model = create_thot_word_alignment_model(self._word_alignment_model_type)
        self._symmetrized_word_alignment_model = ThotSymmetrizedWordAlignmentModel(
            self._direct_word_alignment_model, self._inverse_word_alignment_model
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

    def get_word_graph(self, segment: Union[str, Sequence[str]]) -> WordGraph:
        source_tokens = self._tokenize_source(segment)
        normalized_source_tokens = self._normalize_source(source_tokens)

        decoder = self._get_decoder()
        word_graph = decoder.get_word_graph(to_sentence(normalized_source_tokens))
        return self._create_word_graph(source_tokens, normalized_source_tokens, word_graph)

    def translate(self, segment: Union[str, Sequence[str]]) -> TranslationResult:
        source_tokens = self._tokenize_source(segment)
        normalized_source_tokens = self._normalize_source(source_tokens)

        decoder = self._get_decoder()
        translation = decoder.translate(to_sentence(normalized_source_tokens))
        return self._create_result(source_tokens, normalized_source_tokens, translation)

    def translate_n(self, n: int, segment: Union[str, Sequence[str]]) -> Sequence[TranslationResult]:
        source_tokens = self._tokenize_source(segment)
        normalized_source_tokens = self._normalize_source(source_tokens)

        decoder = self._get_decoder()
        n_translations = decoder.translate_n(to_sentence(normalized_source_tokens), n)
        return [self._create_result(source_tokens, normalized_source_tokens, t) for t in n_translations]

    def translate_batch(self, segments: Sequence[Union[str, Sequence[str]]]) -> Sequence[TranslationResult]:
        preprocessed_segments: List[Tuple[Sequence[str], Sequence[str]]] = []
        for segment in segments:
            source_tokens = self._tokenize_source(segment)
            normalized_source_tokens = self._normalize_source(source_tokens)
            preprocessed_segments.append((source_tokens, normalized_source_tokens))

        decoder = self._get_decoder()
        translations = decoder.translate_batch([to_sentence(nst) for _, nst in preprocessed_segments])
        return [self._create_result(st, nst, td) for (st, nst), td in zip(preprocessed_segments, translations)]

    def translate_n_batch(
        self, n: int, segments: Sequence[Union[str, Sequence[str]]]
    ) -> Sequence[Sequence[TranslationResult]]:
        preprocessed_segments: List[Tuple[Sequence[str], Sequence[str]]] = []
        for segment in segments:
            source_tokens = self._tokenize_source(segment)
            normalized_source_tokens = self._normalize_source(source_tokens)
            preprocessed_segments.append((source_tokens, normalized_source_tokens))

        decoder = self._get_decoder()
        all_translations = decoder.translate_n_batch([to_sentence(nst) for _, nst in preprocessed_segments], n)
        return [
            [self._create_result(st, nst, td) for td in n_translations]
            for (st, nst), n_translations in zip(preprocessed_segments, all_translations)
        ]

    def train_segment(
        self, source_segment: Union[str, Sequence[str]], target_segment: Sequence[str], sentence_start: bool = True
    ) -> None:
        normalized_source_tokens = self._normalize_source(self._tokenize_source(source_segment))
        target_tokens = self._tokenize_target(target_segment)
        normalized_target_tokens = self._normalize_target(target_tokens)

        decoder = self._get_decoder()
        decoder.train_sentence_pair(to_sentence(normalized_source_tokens), to_sentence(normalized_target_tokens))
        if self.truecaser is not None:
            self.truecaser.train_segment(target_tokens, sentence_start)

    def create_trainer(self, corpus: ParallelTextCorpus) -> ThotSmtModelTrainer:
        trainer = _Trainer(self, corpus, self._parameters if self._config_filename is None else self._config_filename)
        trainer.source_tokenizer = self.source_tokenizer
        trainer.target_tokenizer = self.target_tokenizer
        trainer.lowercase_source = self.lowercase_source
        trainer.lowercase_target = self.lowercase_target
        return trainer

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

    def _create_word_graph(
        self, source_tokens: Sequence[str], normalized_source_tokens: Sequence[str], thot_word_graph: tt.WordGraph
    ) -> WordGraph:
        confidence_estimator = Ibm1WordConfidenceEstimator(
            self._symmetrized_word_alignment_model.get_translation_score, normalized_source_tokens
        )

        arcs: List[WordGraphArc] = []
        for thot_arc_id in range(thot_word_graph.num_arcs):
            thot_arc = thot_word_graph.get_arc(thot_arc_id)
            src_phrase_len = thot_arc.source_end_index - thot_arc.source_start_index
            source_segment_range = Range[int].create(thot_arc.source_start_index, thot_arc.source_end_index + 1)
            normalized_tokens: List[str] = []
            confidences: List[float] = []
            for word in thot_arc.words:
                normalized_token = unescape_token(word)
                normalized_tokens.append(normalized_token)
                confidences.append(confidence_estimator.estimate(source_segment_range, normalized_token))
            if src_phrase_len == 1 and len(normalized_tokens) == 1:
                wa_matrix = WordAlignmentMatrix.from_word_pairs(1, 1, {(0, 0)})
            else:
                wa_matrix = self._word_aligner.align(
                    normalized_source_tokens[thot_arc.source_start_index : thot_arc.source_end_index], normalized_tokens
                )

            arcs.append(
                WordGraphArc(
                    thot_arc.in_state,
                    thot_arc.out_state,
                    thot_arc.score,
                    self._denormalize_target(normalized_tokens),
                    wa_matrix,
                    Range[int].create(thot_arc.source_start_index, thot_arc.source_end_index + 1),
                    [TranslationSources.NONE if thot_arc.is_unknown else TranslationSources.SMT]
                    * len(normalized_tokens),
                    confidences,
                )
            )
        word_graph = WordGraph(source_tokens, arcs, thot_word_graph.final_states, thot_word_graph.initial_state_score)
        return word_graph

    def _create_result(
        self, source_tokens: Sequence[str], normalized_source_tokens: Sequence[str], data: tt.TranslationData
    ) -> TranslationResult:
        normalized_target_tokens = to_target_tokens(data.target)
        target_tokens = self._denormalize_target(normalized_target_tokens)

        builder = TranslationResultBuilder(source_tokens, self.target_detokenizer)
        confidence_estimator = Ibm1WordConfidenceEstimator(
            self._symmetrized_word_alignment_model.get_translation_score, normalized_source_tokens
        )
        trg_phrase_start_index = 0
        for k in range(len(data.source_segmentation)):
            source_start_index, source_end_index = data.source_segmentation[k]
            source_start_index -= 1
            target_cut = data.target_segment_cuts[k]

            source_segment_range = Range[int].create(source_start_index, source_end_index)
            for j in range(trg_phrase_start_index, target_cut):
                builder.append_token(
                    target_tokens[j],
                    TranslationSources.NONE if j in data.target_unknown_words else TranslationSources.SMT,
                    confidence_estimator.estimate(source_segment_range, normalized_target_tokens[j]),
                )

            src_phrase_len = source_end_index - source_start_index
            trg_phrase_len = target_cut - trg_phrase_start_index
            if src_phrase_len == 1 and trg_phrase_len == 1:
                wa_matrix = WordAlignmentMatrix.from_word_pairs(1, 1, {(0, 0)})
            else:
                src_phrase = [""] * src_phrase_len
                for i in range(src_phrase_len):
                    src_phrase[i] = normalized_source_tokens[source_start_index + i]
                trg_phrase = [""] * trg_phrase_len
                for j in range(trg_phrase_len):
                    trg_phrase[j] = normalized_target_tokens[trg_phrase_start_index + j]
                wa_matrix = self._word_aligner.align(src_phrase, trg_phrase)
            builder.mark_phrase(source_segment_range, wa_matrix)
            trg_phrase_start_index += trg_phrase_len

        result = builder.to_result()
        return result

    def _load(self) -> None:
        self._model = load_smt_model(self._word_alignment_model_type, self._parameters)
        self._direct_word_alignment_model._set_model(self._model.direct_word_alignment_model, owned=True)
        self._inverse_word_alignment_model._set_model(self._model.inverse_word_alignment_model, owned=True)
        self._symmetrized_word_alignment_model._reset_aligner()

    def _tokenize_source(self, source_segment: Union[str, Sequence[str]]) -> Sequence[str]:
        if isinstance(source_segment, str):
            return list(self.source_tokenizer.tokenize(source_segment))
        return source_segment

    def _normalize_source(self, source_tokens: Sequence[str]) -> Sequence[str]:
        if self.lowercase_source:
            return lowercase(source_tokens)
        return source_tokens

    def _tokenize_target(self, target_segment: Union[str, Sequence[str]]) -> Sequence[str]:
        if isinstance(target_segment, str):
            return list(self.target_tokenizer.tokenize(target_segment))
        return target_segment

    def _normalize_target(self, target_tokens: Sequence[str]) -> Sequence[str]:
        if self.lowercase_target:
            return lowercase(target_tokens)
        return target_tokens

    def _detokenize_target(self, target_tokens: Sequence[str]) -> str:
        return self.target_detokenizer.detokenize(target_tokens)

    def _denormalize_target(self, normalized_target_tokens: Sequence[str]) -> Sequence[str]:
        if self.truecaser is None:
            return normalized_target_tokens
        return self.truecaser.truecase(normalized_target_tokens)

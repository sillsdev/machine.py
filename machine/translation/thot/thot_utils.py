from typing import Iterable, List, Sequence, Tuple

import thot.alignment as ta
import thot.translation as tt

from .thot_smt_parameters import ThotSmtParameters
from .thot_word_alignment_model_type import ThotWordAlignmentModelType


def batch(
    segments: Iterable[Sequence[Sequence[str]]], batch_size: int
) -> Iterable[Tuple[List[Sequence[str]], List[Sequence[str]]]]:
    src_segments: List[Sequence[str]] = []
    trg_segments: List[Sequence[str]] = []
    for source_segment, target_segment in segments:
        src_segments.append(list(escape_tokens(source_segment)))
        trg_segments.append(list(escape_tokens(target_segment)))
        if len(src_segments) == batch_size:
            yield src_segments, trg_segments
            src_segments.clear()
            trg_segments.clear()
    if len(src_segments) > 0:
        yield src_segments, trg_segments


def load_smt_model(word_alignment_model_type: ThotWordAlignmentModelType, parameters: ThotSmtParameters) -> tt.SmtModel:
    if word_alignment_model_type == ThotWordAlignmentModelType.IBM1:
        model_type = ta.AlignmentModelType.INCR_IBM1
    elif word_alignment_model_type == ThotWordAlignmentModelType.IBM2:
        model_type = ta.AlignmentModelType.INCR_IBM2
    elif word_alignment_model_type == ThotWordAlignmentModelType.HMM:
        model_type = ta.AlignmentModelType.INCR_HMM
    elif word_alignment_model_type == ThotWordAlignmentModelType.FAST_ALIGN:
        model_type = ta.AlignmentModelType.FAST_ALIGN
    elif word_alignment_model_type == ThotWordAlignmentModelType.IBM3:
        model_type = ta.AlignmentModelType.IBM3
    elif word_alignment_model_type == ThotWordAlignmentModelType.IBM4:
        model_type = ta.AlignmentModelType.IBM4

    model = tt.SmtModel(model_type)
    model.load_translation_model(parameters.translation_model_filename_prefix)
    model.load_language_model(parameters.language_model_filename_prefix)
    model.non_monotonicity = parameters.model_non_monotonicity
    model.w = parameters.model_w
    model.a = parameters.model_a
    model.e = parameters.model_e
    model.heuristic = parameters.model_heuristic
    olp = tt.OnlineTrainingParameters()
    olp.algorithm = parameters.learning_algorithm
    olp.learning_rate_policy = parameters.learning_rate_policy
    olp.learn_step_size = parameters.learning_step_size
    olp.em_iters = parameters.learning_em_iters
    olp.e = parameters.learning_e
    olp.r = parameters.learning_r
    model.online_training_parameters = olp
    if len(parameters.model_weights) > 0:
        model.weights = parameters.model_weights
    return model


def load_smt_decoder(model: tt.SmtModel, parameters: ThotSmtParameters) -> tt.SmtDecoder:
    decoder = tt.SmtDecoder(model)
    decoder.s = parameters.decoder_s
    decoder.is_breadth_first = parameters.decoder_breadth_first
    return decoder


def escape_token(token: str) -> str:
    if token == "|||":
        return "<3bars>"
    return token


def escape_tokens(segment: Iterable[str]) -> Iterable[str]:
    return (escape_token(t) for t in segment)


def unescape_token(token: str) -> str:
    if token == "<3bars>":
        return "|||"
    return token


def unescape_tokens(segment: Iterable[str]) -> Iterable[str]:
    return (unescape_token(t) for t in segment)

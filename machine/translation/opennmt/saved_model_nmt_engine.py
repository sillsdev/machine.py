from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Union

import tensorflow as tf

from ...annotations import Range
from ...tokenization.detokenizer import Detokenizer
from ...tokenization.tokenizer import Tokenizer
from ...tokenization.whitespace_detokenizer import WHITESPACE_DETOKENIZER
from ...tokenization.whitespace_tokenizer import WHITESPACE_TOKENIZER
from ...utils.typeshed import StrPath
from ..translation_engine import TranslationEngine
from ..translation_result import TranslationResult
from ..translation_result_builder import TranslationResultBuilder
from ..translation_sources import TranslationSources
from ..word_alignment_matrix import WordAlignmentMatrix


@dataclass
class SavedModelTranslateSignature:
    signature_key: str = "serving_default"
    input_tokens_key: str = "tokens"
    input_length_key: str = "length"
    input_ref_key: str = "ref"
    input_ref_length_key: str = "ref_length"
    output_tokens_key: str = "tokens"
    output_length_key: str = "length"
    output_alignment_key: str = "alignment"


class SavedModelNmtEngine(TranslationEngine):
    def __init__(
        self,
        model_filename: StrPath,
        signature: SavedModelTranslateSignature = SavedModelTranslateSignature(),
        source_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        target_detokenizer: Detokenizer[str, str] = WHITESPACE_DETOKENIZER,
    ) -> None:
        self._signature = signature
        self._model: Any = tf.saved_model.load(str(model_filename))
        self._translate_fn = self._model.signatures[signature.signature_key]
        self.source_tokenizer = source_tokenizer
        self.target_detokenizer = target_detokenizer

    def translate(self, segment: Union[str, Sequence[str]]) -> TranslationResult:
        return next(iter(self.translate_n(1, segment)))

    def translate_n(self, n: int, segment: Union[str, Sequence[str]]) -> Iterable[TranslationResult]:
        if isinstance(segment, str):
            source_tokens = list(self.source_tokenizer.tokenize(segment))
        else:
            source_tokens = segment

        inputs = {
            self._signature.input_tokens_key: tf.constant([source_tokens], dtype=tf.string),
            self._signature.input_length_key: tf.constant([len(source_tokens)], dtype=tf.int32),
            self._signature.input_ref_key: tf.constant([[""]], dtype=tf.string),
            self._signature.input_ref_length_key: tf.constant([1], dtype=tf.int32),
        }
        outputs = self._translate_fn(**inputs)
        output_tokens = outputs[self._signature.output_tokens_key]
        output_lengths = outputs[self._signature.output_length_key]
        output_alignments = outputs[self._signature.output_alignment_key]
        output_count = output_lengths.shape[0]
        i = 0
        while i < n or i < output_count:
            output_length_i = int(output_lengths[0][i].numpy())
            output_tokens_i = output_tokens[0][i][:output_length_i]
            builder = TranslationResultBuilder(source_tokens, self.target_detokenizer)
            for word in output_tokens_i.numpy():
                builder.append_token(word.decode("utf-8"), TranslationSources.NMT, -1)

            alignment = output_alignments[0][i]
            src_indices = tf.argmax(alignment[:output_length_i], axis=-1).numpy()
            wa_matrix = WordAlignmentMatrix.from_word_pairs(
                len(source_tokens), output_length_i, set(zip(src_indices, range(output_length_i)))
            )
            builder.mark_phrase(Range.create(0, len(segment)), wa_matrix)

            yield builder.to_result()
            i += 1

    def translate_batch(self, segments: Sequence[Union[str, Sequence[str]]]) -> Sequence[TranslationResult]:
        return [results[0] for results in self.translate_n_batch(1, segments)]

    def translate_n_batch(
        self, n: int, segments: Sequence[Union[str, Sequence[str]]]
    ) -> Sequence[Sequence[TranslationResult]]:
        raise NotImplementedError

    def __enter__(self) -> SavedModelNmtEngine:
        return self
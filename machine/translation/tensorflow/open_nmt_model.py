import heapq
from types import TracebackType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type, cast

import numpy as np
import tensorflow as tf
from opennmt import Runner
from opennmt.config import load_model_from_catalog
from opennmt.data import inference_pipeline
from opennmt.inputters import Inputter
from opennmt.models import SequenceToSequence
from opennmt.utils.checkpoint import Checkpoint
from opennmt.utils.misc import extract_batches, item_or_tuple

from ...annotations.range import Range
from ...corpora.parallel_text_corpus_view import ParallelTextCorpusView
from ..trainer import Trainer
from ..translation_engine import TranslationEngine
from ..translation_model import TranslationModel
from ..translation_result import TranslationResult
from ..translation_result_builder import TranslationResultBuilder
from ..translation_sources import TranslationSources
from ..word_alignment_matrix import WordAlignmentMatrix


class OpenNmtModel(TranslationModel, Runner):
    def __init__(
        self,
        model_type: str,
        config: dict,
        mixed_precision: bool = False,
    ):
        super().__init__(load_model_from_catalog(model_type), config, mixed_precision=mixed_precision)
        self._engines: Set[OpenNmtEngine] = set()

    @property
    def config(self) -> dict:
        return self._config

    def create_engine(self) -> "OpenNmtEngine":
        engine = OpenNmtEngine(self)
        self._engines.add(engine)
        return engine

    def create_trainer(self, corpus: ParallelTextCorpusView) -> Trainer:
        ...

    def restore_checkpoint(self) -> Tuple[SequenceToSequence, dict]:
        config = self._finalize_config()
        model = self._init_model(config)
        checkpoint = Checkpoint.from_config(config, model)
        checkpoint.restore(weights_only=True)
        return model, config

    def remove_engine(self, engine: "OpenNmtEngine") -> None:
        self._engines.remove(engine)

    def __enter__(self) -> "OpenNmtModel":
        return self

    def _finalize_config(self, training=False, num_replicas=1, num_devices=1):
        config = super()._finalize_config(training, num_replicas, num_devices)
        config["params"]["num_hypotheses"] = 0
        return config


class OpenNmtEngine(TranslationEngine):
    def __init__(self, model: OpenNmtModel):
        self._model = model
        self._checkpoint_model, self._config = self._model.restore_checkpoint()
        self._infer_fn: Any = None

    @property
    def _features_inputter(self) -> Inputter:
        return cast(Inputter, self._checkpoint_model.features_inputter)

    @property
    def _labels_inputter(self) -> Inputter:
        return cast(Inputter, self._checkpoint_model.labels_inputter)

    def translate(self, segment: Sequence[str]) -> TranslationResult:
        return self.translate_n(1, segment)[0]

    def translate_n(self, n: int, segment: Sequence[str]) -> List[TranslationResult]:
        return next(iter(self.translate_many_n(n, [segment])))

    def translate_many(self, segments: Iterable[Sequence[str]]) -> Iterable[TranslationResult]:
        return (hyps[0] for hyps in self.translate_many_n(1, segments))

    def translate_many_n(self, n: int, segments: Iterable[Sequence[str]]) -> Iterable[List[TranslationResult]]:
        infer_config = self._config["infer"]
        features = (" ".join(segment) for segment in segments)
        dataset = self._make_inference_dataset(
            [features],
            infer_config["batch_size"],
            length_bucket_width=infer_config["length_bucket_width"],
            prefetch_buffer_size=infer_config.get("prefetch_buffer_size"),
        )

        if self._infer_fn is None:
            self._infer_fn = tf.function(self._checkpoint_model.infer, input_signature=(dataset.element_spec,))
            if not tf.config.functions_run_eagerly():
                tf.get_logger().info("Tracing and optimizing the inference graph...")
                self._infer_fn.get_concrete_function()  # Trace the function now.

        queued_results: Dict[int, List[TranslationResult]] = {}
        next_index = 0
        heap: List[int] = []
        for source in dataset:
            predictions = self._infer_fn(source)
            predictions["src_tokens"] = source["tokens"]
            predictions["src_length"] = source["length"]
            predictions = tf.nest.map_structure(lambda t: t.numpy(), predictions)
            for prediction in extract_batches(predictions):
                index: int = prediction["index"]

                src_length = prediction["src_length"]
                src_tokens = prediction["src_tokens"][:src_length]
                src_segment = self._features_inputter.tokenizer.detokenize(src_tokens)

                num_hypotheses = min(n, len(prediction["log_probs"]))
                hypotheses: List[TranslationResult] = []
                for i in range(num_hypotheses):
                    trg_length = prediction["length"][i]
                    trg_tokens = prediction["tokens"][i][:trg_length]
                    trg_segment = self._labels_inputter.tokenizer.detokenize(trg_tokens)

                    builder = TranslationResultBuilder()

                    alignment = prediction["alignment"][i][:trg_length]
                    src_indices = np.argmax(alignment, axis=-1)
                    confidences = np.max(alignment, axis=-1)
                    trg_indices = range(trg_length)
                    wa_matrix = WordAlignmentMatrix.from_word_pairs(
                        src_length, trg_length, set(zip(src_indices, trg_indices))
                    )

                    for token, confidence in zip(trg_segment.split(" "), confidences):
                        builder.append_word(token, TranslationSources.NMT, confidence)

                    builder.mark_phrase(Range.create(0, src_length), wa_matrix)
                    hypotheses.append(builder.to_result(src_segment.split(" ")))

                queued_results[index] = hypotheses
                heapq.heappush(heap, index)
                while len(heap) > 0 and heap[0] == next_index:
                    i = heapq.heappop(heap)
                    result = queued_results.pop(i)
                    yield result
                    next_index += 1

    def __enter__(self) -> "OpenNmtEngine":
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self._model.remove_engine(self)
        return None

    def _make_inference_dataset(
        self,
        features_list: List[Iterable[str]],
        batch_size: int,
        batch_type: str = "examples",
        length_bucket_width: Optional[int] = None,
        num_threads: int = 1,
        prefetch_buffer_size: Optional[int] = None,
    ):
        def _map_fn(*arg):
            features = self._features_inputter.make_features(element=item_or_tuple(arg), training=False)
            if isinstance(features, (list, tuple)):
                # Special case for unsupervised inputters that always return a
                # tuple (features, labels).
                return features[0]
            return features

        transform_fns = [lambda dataset: dataset.map(_map_fn, num_parallel_calls=num_threads or 1)]

        def _gen(index: int):
            for features in features_list[index]:
                yield features

        spec = tf.TensorSpec(shape=(), dtype=tf.string)  # type: ignore
        if len(features_list) == 1:
            dataset = tf.data.Dataset.from_generator(_gen, args=(0,), output_signature=spec)
        else:
            datasets = tuple(
                tf.data.Dataset.from_generator(_gen, args=(i,), output_signature=spec)
                for i in range(len(features_list))
            )
            dataset = tf.data.Dataset.zip(datasets)
        dataset = dataset.apply(
            inference_pipeline(
                batch_size,
                batch_type=batch_type,
                transform_fns=transform_fns,
                length_bucket_width=length_bucket_width,
                length_fn=self._features_inputter.get_length,
                num_threads=num_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
        return dataset
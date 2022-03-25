import math
import os
from typing import Callable, Optional

import tensorflow as tf
from opennmt import Runner
from opennmt.config import load_model_from_catalog
from opennmt.evaluation import Evaluator
from opennmt.training import Trainer as OnmtTrainer
from opennmt.utils.checkpoint import Checkpoint
from opennmt.utils.misc import disable_mixed_precision, enable_mixed_precision

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...utils.progress_status import ProgressStatus
from ..trainer import Trainer, TrainStats


class OpenNmtModelTrainer(Runner, Trainer):
    def __init__(
        self,
        model_type: str,
        config: dict,
        corpus: ParallelTextCorpus,
        mixed_precision: bool = False,
    ):
        super().__init__(load_model_from_catalog(model_type), config, mixed_precision=mixed_precision)
        self._corpus = corpus
        self._stats = TrainStats()

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        config = self._finalize_config(training=True, num_replicas=1, num_devices=1)

        mixed_precision = self._mixed_precision and enable_mixed_precision()
        model = self._init_model(config)
        optimizer = model.get_optimizer()

        data_config = config["data"]
        train_config = config["train"]
        eval_config = config["eval"]

        batch_type = train_config["batch_type"]
        batch_size_multiple = 8 if mixed_precision and batch_type == "tokens" else 1

        def dataset_fn(input_context):

            model.examples_inputter.make_training_dataset(
                data_config["train_features_file"],
                data_config.get("train_labels_file"),
                train_config["batch_size"],
                batch_type=batch_type,
                batch_size_multiple=batch_size_multiple,
                shuffle_buffer_size=train_config["sample_buffer_size"],
                length_bucket_width=train_config["length_bucket_width"],
                maximum_features_length=train_config.get("maximum_features_length"),
                maximum_labels_length=train_config.get("maximum_labels_length"),
                single_pass=train_config.get("single_pass", False),
                num_shards=input_context.num_input_pipelines,
                shard_index=input_context.input_pipeline_id,
                prefetch_buffer_size=train_config.get("prefetch_buffer_size"),
                cardinality_multiple=input_context.num_replicas_in_sync,
                weights=data_config.get("train_files_weights"),
                batch_autotune_mode=train_config.get("batch_autotune_mode"),
            )

        checkpoint = Checkpoint.from_config(config, model, optimizer=optimizer)
        checkpoint.restore()
        evaluator = Evaluator.from_config(model, config)

        # Set gradients accumulation based on the requested effective batch size.
        if train_config.get("effective_batch_size") is not None:
            accum_steps = _count_batch_accum(
                train_config["batch_size"],
                train_config["effective_batch_size"],
                num_replicas=1,
            )
            tf.get_logger().info(
                "Accumulate gradients of %d iterations to reach effective batch size of %d",
                accum_steps,
                train_config["effective_batch_size"],
            )
        else:
            accum_steps = 1

        trainer = OnmtTrainer(model, optimizer, checkpoint=checkpoint)

        trainer(
            dataset_fn,
            max_step=train_config.get("max_step"),
            accum_steps=accum_steps,
            report_steps=train_config.get("save_summary_steps", 100),
            save_steps=train_config.get("save_checkpoints_steps", 5000),
            evaluator=evaluator,
            eval_steps=eval_config.get("steps", 5000),
            moving_average_decay=train_config.get("moving_average_decay"),
        )

        average_last_checkpoints = train_config.get("average_last_checkpoints", 0)
        if average_last_checkpoints > 0:
            self.average_checkpoints(
                os.path.join(checkpoint.model_dir, "avg"),
                max_count=average_last_checkpoints,
            )

        if mixed_precision:
            disable_mixed_precision()

        if evaluator.last_evaluated_step is not None:
            for name, value in evaluator.last_evaluated_step:
                self._stats.metrics[name] = value

    def save(self) -> None:
        ...

    @property
    def stats(self) -> TrainStats:
        return self._stats


def _count_batch_accum(batch_size, target_batch_size, num_replicas=1):
    """Given the current batch size, the number of replicas, and the requested
    effective batch size, returns the number of gradients to accumulate.
    """
    return int(math.ceil(float(target_batch_size) / (batch_size * num_replicas)))

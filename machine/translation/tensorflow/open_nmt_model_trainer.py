import math
import os
import shutil
from contextlib import ExitStack
from typing import Callable, Optional

import opennmt.training as onmt_training
import tensorflow as tf
from opennmt import Runner
from opennmt.config import load_model_from_catalog
from opennmt.evaluation import Evaluator
from opennmt.models import Model
from opennmt.utils.checkpoint import Checkpoint
from opennmt.utils.misc import disable_mixed_precision, enable_mixed_precision

from ...corpora.corpora_utils import get_split_indices
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
        resume: bool = True,
    ):
        super().__init__(load_model_from_catalog(model_type), config, mixed_precision=mixed_precision)
        self._corpus = corpus
        self._resume = resume
        self._stats = TrainStats()
        self.val_size = 250

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        config = self._finalize_config(training=True, num_replicas=1, num_devices=1)

        mixed_precision = self._mixed_precision and enable_mixed_precision()
        model: Model = self._init_model(config)
        optimizer = model.get_optimizer()

        data_config = config["data"]
        train_config = config["train"]
        eval_config = config["eval"]

        batch_type = train_config["batch_type"]
        batch_size_multiple = 8 if mixed_precision and batch_type == "tokens" else 1

        train_src_path = data_config["train_features_file"]
        train_trg_path = data_config["train_labels_file"]
        eval_src_path = data_config["eval_features_file"]
        eval_trg_path = data_config["eval_labels_file"]

        max_features_length = train_config.get("maximum_features_length")
        max_labels_length = train_config.get("maximum_labels_length")

        corpus = self._corpus.filter(
            lambda row: not row.is_empty
            and (max_features_length is None or len(row.source_segment) <= max_features_length)
            and (max_labels_length is None or len(row.target_segment) <= max_labels_length)
        )
        corpus_size = corpus.count()

        if not self._resume:
            if os.path.isdir(self.model_dir):
                shutil.rmtree(self.model_dir)
            if os.path.isfile(train_src_path):
                os.remove(train_src_path)
            if os.path.isfile(train_trg_path):
                os.remove(train_trg_path)
            if os.path.isfile(eval_src_path):
                os.remove(eval_src_path)
            if os.path.isfile(eval_trg_path):
                os.remove(eval_trg_path)

        if (
            not os.path.isfile(train_src_path)
            or not os.path.isfile(train_trg_path)
            or not os.path.isfile(eval_src_path)
            or not os.path.isfile(eval_trg_path)
        ):
            test_indices = get_split_indices(corpus_size, size=self.val_size, seed=31415)
            with ExitStack() as stack:
                train_src_file = stack.enter_context(open(train_src_path, "w", encoding="utf-8", newline="\n"))
                train_trg_file = stack.enter_context(open(train_trg_path, "w", encoding="utf-8", newline="\n"))
                eval_src_file = stack.enter_context(open(eval_src_path, "w", encoding="utf-8", newline="\n"))
                eval_trg_file = stack.enter_context(open(eval_trg_path, "w", encoding="utf-8", newline="\n"))

                for i, row in enumerate(corpus):
                    if i in test_indices:
                        eval_src_file.write(row.source_text + "\n")
                        eval_trg_file.write(row.target_text + "\n")
                    else:
                        train_src_file.write(row.source_text + "\n")
                        train_trg_file.write(row.target_text + "\n")

        def dataset_fn(input_context):
            return model.examples_inputter.make_training_dataset(
                train_src_path,
                train_trg_path,
                train_config["batch_size"],
                batch_type=batch_type,
                batch_size_multiple=batch_size_multiple,
                shuffle_buffer_size=train_config["sample_buffer_size"],
                length_bucket_width=train_config["length_bucket_width"],
                maximum_features_length=max_features_length,
                maximum_labels_length=max_labels_length,
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

        trainer = _Trainer(progress, check_canceled, model, optimizer, checkpoint=checkpoint)

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

        self._stats.trained_segment_count = max(corpus_size - self.val_size, 0)
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


class _Trainer(onmt_training.Trainer):
    def __init__(
        self,
        progress: Optional[Callable[[ProgressStatus], None]],
        check_canceled: Optional[Callable[[], None]],
        model,
        optimizer,
        checkpoint=None,
    ):
        super().__init__(model, optimizer, checkpoint)
        self._progress = progress
        self._check_canceled = check_canceled

    def __call__(
        self,
        dataset,
        max_step=None,
        accum_steps=1,
        report_steps=100,
        save_steps=5000,
        evaluator=None,
        eval_steps=5000,
        moving_average_decay=None,
    ):
        if max_step is not None and self._optimizer.iterations.numpy() >= max_step:
            raise RuntimeError(
                "The training already reached max_step (%d). If you "
                "want to continue the training, you should increase the "
                "max_step value in the training parameters." % max_step
            )
        if evaluator is not None and evaluator.should_stop():
            raise RuntimeError(
                "The early stopping conditions are already met. If you "
                "want to continue the training, you should update your "
                "early stopping parameters."
            )

        self._gradient_accumulator.reset()

        with self._summary_writer.as_default():
            self._training_stats = onmt_training.TrainingStats(
                self._model, self._optimizer, reduce_fn=self._all_reduce_sum
            )
            iterations = self._optimizer.iterations
            tf.summary.experimental.set_step(iterations)  # type: ignore

            step = None
            moving_average = None
            for i, loss in enumerate(self._steps(dataset, accum_steps=accum_steps, report_steps=report_steps)):
                if i == 0:
                    self._log_model_info()

                if moving_average_decay is not None and self.is_master:
                    if moving_average is None:
                        moving_average = onmt_training.MovingAverage(
                            self._model.trainable_variables, iterations, decay=moving_average_decay
                        )
                    self._update_moving_average(moving_average)

                step = iterations.numpy()

                reset_throughput = False
                self._training_stats.update_on_step(step, loss)
                if step % report_steps == 0:
                    self._training_stats.log(self.is_master)
                    reset_throughput = True
                    if self._check_canceled is not None:
                        self._check_canceled()
                    if self._progress is not None:
                        summary = self._training_stats.get_last_summary()
                        steps_per_sec = summary["steps_per_sec"]
                        words_per_sec_str = ""
                        for name, avg in sorted(summary["words_per_sec"].items(), key=lambda x: x[0]):
                            words_per_sec_str += f", {name} words/s = {avg}"
                        learning_rate = summary["learning_rate"]
                        message = (
                            f"Step = {step} ; steps/s = {steps_per_sec:0.2f}{words_per_sec_str} ; "
                            f"Learning rate = {learning_rate} ; Loss = {loss}"
                        )
                        self._progress(ProgressStatus(step, message=message))
                if step == 1 or (save_steps is not None and step % save_steps == 0):
                    self._save_checkpoint(step, moving_average=moving_average)
                    reset_throughput = True
                if eval_steps is not None and step % eval_steps == 0:
                    early_stop = self._evaluate(evaluator, step, moving_average=moving_average)
                    reset_throughput = True
                    if early_stop:
                        tf.get_logger().warning("Early stopping conditions are met. Exiting.")
                        break
                if step == max_step:
                    break
                if reset_throughput:
                    self._training_stats.reset_throughput()

            if step is None:
                raise RuntimeError(
                    "No training steps were executed. This usually means the "
                    "training file is empty or all examples were filtered out. "
                    "For the latter, verify that the maximum_*_length values are "
                    "consistent with your data."
                )

            self._training_stats.log_final(self.is_master)
            summary = self._training_stats.get_global_summary()
            if self._progress is not None:
                last_step = summary["last_step"]
                last_loss = summary["last_loss"]
                avg_loss = summary["average_loss"]
                message = f"Last step = {last_step} ; Last loss = {last_loss} ; Avg loss = {avg_loss}"
                self._progress(ProgressStatus(last_step, message=message))
            self._save_checkpoint(step, moving_average=moving_average)
            self._evaluate(evaluator, step, moving_average=moving_average)
            return summary

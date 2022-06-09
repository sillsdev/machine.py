import copy
import math
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Optional, cast

import opennmt.training as onmt_training
import tensorflow as tf
from opennmt.config import try_prefix_paths
from opennmt.evaluation import Evaluator
from opennmt.models import Model
from opennmt.utils.checkpoint import Checkpoint
from opennmt.utils.misc import disable_mixed_precision, enable_mixed_precision

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...utils.progress_status import ProgressStatus
from ..trainer import Trainer, TrainStats
from .open_nmt_utils import (
    OpenNmtRunner,
    delete_corpus_files,
    delete_model,
    delete_train_summary_files,
    model_exists,
    move_corpus_files,
    move_model,
)


class OpenNmtModelTrainer(Trainer):
    def __init__(
        self,
        model_type: str,
        config: dict,
        corpus: Optional[ParallelTextCorpus] = None,
        parent_config: Optional[dict] = None,
        mixed_precision: bool = False,
        resume: bool = False,
        val_size: int = 250,
        replace_on_save: bool = True,
    ):
        self._config = config
        self._replace_on_save = replace_on_save
        if self._replace_on_save:
            runner_config: dict = copy.deepcopy(self._config)
            runner_config["data"] = cast(dict, try_prefix_paths(runner_config["model_dir"], runner_config["data"]))
            runner_config["model_dir"] = os.path.join(runner_config["model_dir"], "train.tmp")
            if corpus is not None:
                runner_data_config = runner_config["data"]
                data_config = self._config["data"]
                runner_data_config["train_features_file"] = data_config["train_features_file"]
                runner_data_config["train_labels_file"] = data_config["train_labels_file"]
                runner_data_config["eval_features_file"] = data_config["eval_features_file"]
                runner_data_config["eval_labels_file"] = data_config["eval_labels_file"]
        else:
            runner_config = config

        self._runner = OpenNmtRunner(model_type, runner_config, mixed_precision, prefix_corpus_paths=corpus is not None)
        self._corpus = corpus
        self._parent_config = parent_config
        self.resume = resume
        self._stats = TrainStats()
        self.val_size = val_size

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        delete_train_summary_files(self._runner.model_dir)
        if not self.resume:
            delete_model(self._runner.model_dir)
            if self._corpus is not None:
                delete_corpus_files(self._runner.config)

        mixed_precision = self._runner.mixed_precision and enable_mixed_precision()

        config, model = self._runner.load(training=True)
        optimizer = model.get_optimizer()

        data_config: dict = config["data"]
        train_config: dict = config["train"]
        eval_config: dict = config["eval"]

        train_corpus_size = self._create_corpus_files(train_config, data_config, model)
        parent_checkpoint_path = self._create_parent_checkpoint(data_config)

        checkpoint = Checkpoint.from_config(config, model, optimizer=optimizer)
        checkpoint.restore(parent_checkpoint_path, weights_only=parent_checkpoint_path is not None)
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

        batch_type = train_config["batch_type"]
        batch_size_multiple = 8 if mixed_precision and batch_type == "tokens" else 1

        def dataset_fn(input_context):
            return model.examples_inputter.make_training_dataset(
                data_config["train_features_file"],
                data_config["train_labels_file"],
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
            self._runner.average_checkpoints(
                str(Path(checkpoint.model_dir) / "avg"), max_count=average_last_checkpoints
            )

        if mixed_precision:
            disable_mixed_precision()

        self._stats.train_corpus_size = train_corpus_size
        if evaluator.metrics_history is not None:
            for name, value in evaluator.metrics_history[-1][1].items():
                self._stats.metrics[name] = value

    def save(self) -> None:
        delete_train_summary_files(self._runner.model_dir)
        if self._replace_on_save:
            model_dir = self._config["model_dir"]
            delete_model(model_dir)
            move_model(self._runner.model_dir, model_dir)
            if self._corpus is not None:
                delete_corpus_files(self._config)
                move_corpus_files(self._runner.config, self._config)
            shutil.rmtree(self._runner.model_dir)

    @property
    def stats(self) -> TrainStats:
        return self._stats

    def _create_corpus_files(self, train_config: dict, data_config: dict, model: Model) -> int:
        train_src_path = Path(data_config["train_features_file"])
        train_trg_path = Path(data_config["train_labels_file"])

        if self._corpus is None:
            return model.examples_inputter.get_dataset_size([str(train_src_path), str(train_trg_path)])

        eval_src_path = Path(data_config["eval_features_file"])
        eval_trg_path = Path(data_config["eval_labels_file"])

        if (
            not train_src_path.is_file()
            or not train_trg_path.is_file()
            or not eval_src_path.is_file()
            or not eval_trg_path.is_file()
        ):
            max_src_length: int = train_config["maximum_features_length"]
            max_trg_length: int = train_config["maximum_labels_length"]
            corpus, _, _ = self._corpus.interleaved_split(size=self.val_size, include_empty=False, seed=31415)
            train_corpus_size = 0
            with ExitStack() as stack:
                train_src_path.parent.mkdir(parents=True, exist_ok=True)
                train_src_file = stack.enter_context(train_src_path.open("w", encoding="utf-8", newline="\n"))
                train_trg_path.parent.mkdir(parents=True, exist_ok=True)
                train_trg_file = stack.enter_context(train_trg_path.open("w", encoding="utf-8", newline="\n"))
                eval_src_path.parent.mkdir(parents=True, exist_ok=True)
                eval_src_file = stack.enter_context(eval_src_path.open("w", encoding="utf-8", newline="\n"))
                eval_trg_path.parent.mkdir(parents=True, exist_ok=True)
                eval_trg_file = stack.enter_context(eval_trg_path.open("w", encoding="utf-8", newline="\n"))
                rows = stack.enter_context(corpus)

                for row, is_test_row in rows:
                    if is_test_row:
                        eval_src_file.write(row.source_text + "\n")
                        eval_trg_file.write(row.target_text + "\n")
                    elif len(row.source_segment) <= max_src_length and len(row.target_segment) <= max_trg_length:
                        train_src_file.write(row.source_text + "\n")
                        train_trg_file.write(row.target_text + "\n")
                        train_corpus_size += 1

            return train_corpus_size
        else:
            return model.examples_inputter.get_dataset_size([str(train_src_path), str(train_trg_path)])

    def _create_parent_checkpoint(self, data_config: dict) -> Optional[str]:
        if self._parent_config is None:
            return None

        parent_checkpoint_path = Path(self._runner.model_dir) / "parent"
        if parent_checkpoint_path.is_dir():
            return None if model_exists(self._runner.model_dir) else str(parent_checkpoint_path)
        else:
            parent_runner = OpenNmtRunner(self._runner.model_type, self._parent_config)
            parent_runner.update_vocab(
                str(parent_checkpoint_path), data_config["source_vocabulary"], data_config["target_vocabulary"]
            )
            return str(parent_checkpoint_path)


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
                    early_stop = self._evaluate(evaluator, step.item(), moving_average=moving_average)
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
            self._evaluate(evaluator, step.item(), moving_average=moving_average)
            return summary

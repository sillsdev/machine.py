import logging
from contextlib import ExitStack
from typing import Any, Optional, Sequence

import pandas as pd
from clearml import Task, Model

from ..utils.canceled_error import CanceledError
from ..corpora.corpora_utils import batch
from ..translation.translation_engine import TranslationEngine
from .nmt_model_factory import NmtModelFactory
from .shared_file_service import PretranslationInfo, PretranslationWriter, SharedFileService

logger = logging.getLogger(__name__)


class NmtEngineBuildJob:
    def __init__(self, config: Any, nmt_model_factory: NmtModelFactory, shared_file_service: SharedFileService) -> None:
        self._config = config
        self._nmt_model_factory = nmt_model_factory
        self._shared_file_service = shared_file_service
        self.clearml_task: Optional[Task] = None
        self.inference_iteration = 0

    def run(self, enable_clearml: bool) -> None:
        self.clearml_task = Task.init()

        self._nmt_model_factory.init()

        logger.info("Downloading data files")
        source_corpus = self._shared_file_service.create_source_corpus()
        target_corpus = self._shared_file_service.create_target_corpus()
        parallel_corpus = source_corpus.align_rows(target_corpus)

        self.check_canceled()

        self.update_inference_step(-5)

        if self._nmt_model_factory.train_tokenizer:
            logger.info("Training source tokenizer")
            with self._nmt_model_factory.create_source_tokenizer_trainer(source_corpus) as source_tokenizer_trainer:
                source_tokenizer_trainer.train(check_canceled=self.check_canceled)
                source_tokenizer_trainer.save()

            self.check_canceled()

            logger.info("Training target tokenizer")
            with self._nmt_model_factory.create_target_tokenizer_trainer(target_corpus) as target_tokenizer_trainer:
                target_tokenizer_trainer.train(check_canceled=self.check_canceled)
                target_tokenizer_trainer.save()

            self.check_canceled()

        self.update_inference_step(-4)

        logger.info("Training NMT model")
        with self._nmt_model_factory.create_model_trainer(parallel_corpus) as model_trainer:
            model_trainer.train(check_canceled=self.check_canceled)
            model_trainer.save()

        self.check_canceled()

        self.update_inference_step(-1)

        logger.info("Pretranslating segments")
        with ExitStack() as stack:
            model = stack.enter_context(self._nmt_model_factory.create_engine())
            src_pretranslations = stack.enter_context(self._shared_file_service.get_source_pretranslations())
            writer = stack.enter_context(self._shared_file_service.open_target_pretranslation_writer())
            current_inference_step = 0
            self.update_inference_step(current_inference_step)

            batch_size = self._config["batch_size"]
            for pi_batch in batch(src_pretranslations, batch_size):
                self.check_canceled()
                _translate_batch(model, pi_batch, writer)
                current_inference_step += batch_size
                self.update_inference_step(current_inference_step)

    def check_canceled(self) -> None:
        if self.clearml_task is not None:
            if self.clearml_task.get_status() in {"stopped", "stopping"}:
                raise CanceledError

    def update_inference_step(self, step_num: int) -> None:
        if self.clearml_task is not None:
            self.clearml_task.mark_started(force=True)
            self.clearml_task.register_artifact("inference", pd.DataFrame(data=[step_num]))
            self.inference_iteration += 1


def _translate_batch(
    engine: TranslationEngine,
    batch: Sequence[PretranslationInfo],
    writer: PretranslationWriter,
) -> None:
    source_segments = [pi["translation"] for pi in batch]
    for i, result in enumerate(engine.translate_batch(source_segments)):
        batch[i]["translation"] = result.translation
        writer.write(batch[i])

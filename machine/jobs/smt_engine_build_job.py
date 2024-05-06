import json
import logging
import os
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Callable, Optional, cast

from machine.translation.thot.thot_smt_model_trainer import ThotSmtModelTrainer
from machine.translation.thot.thot_word_alignment_model_type import (
    checkThotWordAlignmentModelType,
    getThotWordAlignmentModelType,
)

from ..translation.thot.thot_smt_model import ThotSmtParameters, ThotWordAlignmentModelType
from ..utils.progress_status import ProgressStatus
from .shared_file_service import SharedFileService

logger = logging.getLogger(__name__)


class SmtEngineBuildJob:
    def __init__(self, config: Any, shared_file_service: SharedFileService) -> None:
        self._config = config
        self._shared_file_service = shared_file_service
        self._model_type = cast(str, self._config.model_type).lower()

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        if check_canceled is not None:
            check_canceled()

        self._check_config()

        with TemporaryDirectory() as temp_dir:

            parameters = ThotSmtParameters(
                translation_model_filename_prefix=os.path.join(temp_dir, "tm", "src_trg"),
                language_model_filename_prefix=os.path.join(temp_dir, "lm", "trg.lm"),
            )

            if check_canceled is not None:
                check_canceled()

            logger.info("Downloading data files")
            source_corpus = self._shared_file_service.create_source_corpus()
            target_corpus = self._shared_file_service.create_target_corpus()
            parallel_corpus = source_corpus.align_rows(target_corpus)
            parallel_corpus_size = parallel_corpus.count(include_empty=False)
            if parallel_corpus_size == 0:
                raise RuntimeError("No parallel corpus data found")

            if check_canceled is not None:
                check_canceled()

            with ThotSmtModelTrainer(
                getThotWordAlignmentModelType(self._model_type), parallel_corpus, parameters
            ) as trainer:
                logger.info("Training Model")
                trainer.train(progress=progress, check_canceled=check_canceled)
                trainer.save()
                parameters = trainer.parameters

            if check_canceled is not None:
                check_canceled()

            # zip temp_dir using gzip
            with NamedTemporaryFile() as temp_zip_file:
                with tarfile.open(temp_zip_file.name, mode="w:gz") as tar:
                    # add the model files
                    tar.add(os.path.join(temp_dir, "tm"), arcname="tm")
                    tar.add(os.path.join(temp_dir, "lm"), arcname="lm")

                self._shared_file_service.save_model(Path(temp_zip_file.name), str(self._config.save_model) + ".tar.gz")

    def _check_config(self):
        if "build_options" in self._config:
            try:
                build_options = json.loads(cast(str, self._config.build_options))
            except ValueError as e:
                raise ValueError("Build options could not be parsed: Invalid JSON") from e
            except TypeError as e:
                raise TypeError(f"Build options could not be parsed: {e}") from e
            self._config.update({self._model_type: build_options})
        self._config.data_dir = os.path.expanduser(cast(str, self._config.data_dir))

        logger.info(f"Config: {self._config.as_dict()}")

        if not checkThotWordAlignmentModelType(self._model_type):
            raise RuntimeError(
                f"The model type of {self._model_type} is invalid.  Only the following models are supported:"
                + ", ".join([model.name for model in ThotWordAlignmentModelType])
            )

        if "save_model" not in self._config:
            raise RuntimeError("The save_model parameter is required for SMT build jobs.")

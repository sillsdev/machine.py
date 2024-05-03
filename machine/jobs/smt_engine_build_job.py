import json
import logging
import os
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Optional, cast

from dynaconf.base import Settings

from ..tokenization import get_tokenizer_detokenizer
from ..translation.thot.thot_smt_model import ThotSmtParameters, ThotWordAlignmentModelType
from ..translation.thot.thot_smt_model_trainer import ThotSmtModelTrainer
from ..translation.thot.thot_word_alignment_model_type import (
    checkThotWordAlignmentModelType,
    getThotWordAlignmentModelType,
)
from ..translation.unigram_truecaser_trainer import UnigramTruecaserTrainer
from ..utils.progress_status import ProgressStatus
from .shared_file_service import SharedFileService

logger = logging.getLogger(__name__)


class SmtEngineBuildJob:
    def __init__(self, config: Settings, shared_file_service: SharedFileService) -> None:
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
        (tokenizer, _) = get_tokenizer_detokenizer(str(self._config.get("tokenizer", default="latin")))
        logger.info(f"Tokenizer used: {type(tokenizer).__name__}")

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
                word_alignment_model_type=getThotWordAlignmentModelType(self._model_type),
                corpus=parallel_corpus,
                config=parameters,
                source_tokenizer=tokenizer,
                target_tokenizer=tokenizer,
            ) as trainer:
                logger.info("Training Model")
                trainer.train(progress=progress, check_canceled=check_canceled)
                trainer.save()
                parameters = trainer.parameters

            with UnigramTruecaserTrainer(
                corpus=target_corpus, model_path=os.path.join(temp_dir, "truecase.txt"), tokenizer=tokenizer
            ) as truecase_trainer:
                logger.info("Training Truecaser")
                truecase_trainer.train(progress=progress, check_canceled=check_canceled)
                truecase_trainer.save()

            if check_canceled is not None:
                check_canceled()

            zip_file_name = ""
            with NamedTemporaryFile("w+b", suffix=".tar.gz", delete=False) as temp_zip_file:
                with tarfile.open(fileobj=temp_zip_file, mode="w:gz") as tar:
                    # add the model files
                    tar.add(os.path.join(temp_dir, "tm"), arcname="tm")
                    tar.add(os.path.join(temp_dir, "lm"), arcname="lm")
                    tar.add(os.path.join(temp_dir, "truecase.txt"), arcname="truecase.txt")
                zip_file_name = temp_zip_file.name

            self._shared_file_service.save_model(Path(zip_file_name), str(self._config.save_model) + ".tar.gz")

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

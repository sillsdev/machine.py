import argparse
import logging
from typing import Callable, Optional

from clearml import Task

from ..utils.progress_status import ProgressStatus
from .build_clearml_helper import (
    ProgressInfo,
    get_clearml_check_canceled,
    get_clearml_progress_caller,
    get_local_progress_caller,
    update_settings,
)
from .config import SETTINGS
from .shared_file_service_factory import SharedFileServiceType
from .smt_engine_build_job import SmtEngineBuildJob
from .smt_model_factory import SmtModelFactory
from .translation_file_service import TranslationFileService

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(str(__package__) + ".build_smt_engine")


def run(args: dict) -> None:
    progress: Callable[[ProgressStatus], None]
    check_canceled: Optional[Callable[[], None]] = None
    task = None
    progress_info = ProgressInfo()
    if args["clearml"]:
        task = Task.init()

        check_canceled = get_clearml_check_canceled(progress_info, task)

        task.reload()

        progress = get_clearml_progress_caller(progress_info, task, logger)

    else:
        progress = get_local_progress_caller(ProgressInfo(), logger)

    try:
        logger.info("SMT Engine Build Job started")
        update_settings(SETTINGS, args, task, logger)

        shared_file_service = TranslationFileService(SharedFileServiceType.CLEARML, SETTINGS)
        smt_model_factory: SmtModelFactory
        if SETTINGS.model_type == "thot":
            from .thot.thot_smt_model_factory import ThotSmtModelFactory

            smt_model_factory = ThotSmtModelFactory(SETTINGS)
        else:
            raise RuntimeError("The model type is invalid.")

        smt_engine_build_job = SmtEngineBuildJob(SETTINGS, smt_model_factory, shared_file_service)
        train_corpus_size, confidence = smt_engine_build_job.run(progress, check_canceled)
        if task is not None:
            task.set_progress(100)
            task.set_user_properties(
                {"type": str, "name": "message", "description": "Build Message", "value": "Completed"}
            )
            task.get_logger().report_single_value(name="train_corpus_size", value=train_corpus_size)
            task.get_logger().report_single_value(name="confidence", value=round(confidence, 4))
        logger.info("Finished")
    except Exception as e:
        if task:
            if task.get_status() == "stopped":
                return
            else:
                task.mark_failed(status_reason=type(e).__name__, status_message=str(e))
        raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an SMT model.")
    parser.add_argument("--model-type", required=True, type=str, help="Model type")
    parser.add_argument("--build-id", required=True, type=str, help="Build id")
    parser.add_argument("--clearml", default=False, action="store_true", help="Initializes a ClearML task")
    parser.add_argument("--build-options", default=None, type=str, help="Build configurations")
    args = parser.parse_args()

    input_args = {k: v for k, v in vars(args).items() if v is not None}

    run(input_args)


if __name__ == "__main__":
    main()

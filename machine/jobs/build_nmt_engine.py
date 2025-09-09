import argparse
import logging
from typing import Callable, Optional

from clearml import Task

from ..utils.canceled_error import CanceledError
from ..utils.progress_status import ProgressStatus
from .build_clearml_helper import report_clearml_progress, update_settings
from .config import SETTINGS
from .nmt_engine_build_job import NmtEngineBuildJob
from .nmt_model_factory import NmtModelFactory
from .shared_file_service_factory import SharedFileServiceType
from .translation_file_service import TranslationFileService

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(str(__package__) + ".build_nmt_engine")


def run(args: dict) -> None:
    progress: Optional[Callable[[ProgressStatus], None]] = None
    check_canceled: Optional[Callable[[], None]] = None
    task: Optional[Task] = None
    if args["clearml"]:
        task = Task.init()

        def clearml_check_canceled() -> None:
            if task.get_status() == "stopped":
                raise CanceledError

        check_canceled = clearml_check_canceled

        def clearml_progress(status: ProgressStatus) -> None:
            if status.percent_completed is not None:
                report_clearml_progress(
                    task=task, percent_completed=round(status.percent_completed * 100), progress_status=status
                )

        progress = clearml_progress

    try:
        logger.info("NMT Engine Build Job started")
        update_settings(SETTINGS, args, task, logger)

        translation_file_service = TranslationFileService(SharedFileServiceType.CLEARML, SETTINGS)
        nmt_model_factory: NmtModelFactory
        if SETTINGS.model_type == "huggingface":
            from .huggingface.hugging_face_nmt_model_factory import HuggingFaceNmtModelFactory

            nmt_model_factory = HuggingFaceNmtModelFactory(SETTINGS)
        else:
            raise RuntimeError("The model type is invalid.")

        job = NmtEngineBuildJob(SETTINGS, nmt_model_factory, translation_file_service)
        train_corpus_size, _ = job.run(progress, check_canceled)
        if task is not None:
            task.get_logger().report_single_value(name="train_corpus_size", value=train_corpus_size)
        logger.info("Finished")
    except Exception as e:
        if task:
            if task.get_status() == "stopped":
                return
            else:
                task.mark_failed(status_reason=type(e).__name__, status_message=str(e))
        raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--model-type", required=True, type=str, help="Model type")
    parser.add_argument("--engine-id", required=True, type=str, help="Engine id")
    parser.add_argument("--build-id", required=True, type=str, help="Build id")
    parser.add_argument("--src-lang", required=True, type=str, help="Source language tag")
    parser.add_argument("--trg-lang", required=True, type=str, help="Target language tag")
    parser.add_argument("--clearml", default=False, action="store_true", help="Initializes a ClearML task")
    parser.add_argument("--build-options", default=None, type=str, help="Build configurations")
    parser.add_argument("--save-model", default=None, type=str, help="Save the model using the specified base name")
    args = parser.parse_args()

    run({k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main()

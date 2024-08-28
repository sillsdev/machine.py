import argparse
import logging
from typing import Callable, Optional

from clearml import Task

from ..utils.progress_status import ProgressStatus
from .async_scheduler import AsyncScheduler
from .build_clearml_helper import (
    ProgressInfo,
    create_runtime_properties,
    get_clearml_check_canceled,
    get_clearml_progress_caller,
    get_local_progress_caller,
    update_runtime_properties,
    update_settings,
)
from .config import SETTINGS
from .shared_file_service_factory import SharedFileServiceType
from .thot.thot_word_alignment_model_factory import ThotWordAlignmentModelFactory
from .word_alignment_build_job import WordAlignmentBuildJob
from .word_alignment_file_service import WordAlignmentFileService
from .word_alignment_model_factory import WordAlignmentModelFactory

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(str(__package__) + ".build_word_alignment_model")


def run(args: dict):
    progress: Callable[[ProgressStatus], None]
    check_canceled: Optional[Callable[[], None]] = None
    task = None
    scheduler: Optional[AsyncScheduler] = None
    progress_info = ProgressInfo()
    if args["clearml"]:
        task = Task.init()
        scheduler = AsyncScheduler()

        check_canceled = get_clearml_check_canceled(progress_info, task)

        task.reload()

        progress = get_clearml_progress_caller(progress_info, task, scheduler, logger)

    else:
        progress = get_local_progress_caller(ProgressInfo(), logger)

    try:
        logger.info("Word Alignment Build Job started")
        update_settings(SETTINGS, args)

        logger.info(f"Config: {SETTINGS.as_dict()}")

        word_alignment_file_service = WordAlignmentFileService(SharedFileServiceType.CLEARML, SETTINGS)
        word_alignment_model_factory: WordAlignmentModelFactory
        if SETTINGS.model_type == "thot":
            word_alignment_model_factory = ThotWordAlignmentModelFactory(SETTINGS)
        else:
            raise RuntimeError("The model type is invalid.")

        word_alignment_build_job = WordAlignmentBuildJob(
            SETTINGS, word_alignment_model_factory, word_alignment_file_service
        )
        train_corpus_size = word_alignment_build_job.run(progress, check_canceled)
        if scheduler is not None and task is not None:
            scheduler.schedule(
                update_runtime_properties(
                    task.id, task.session.host, task.session.token, create_runtime_properties(task, 100, "Completed")
                )
            )
            task.get_logger().report_single_value(name="train_corpus_size", value=train_corpus_size)
        logger.info("Finished")
    except Exception as e:
        if task:
            if task.get_status() == "stopped":
                return
            else:
                task.mark_failed(status_reason=type(e).__name__, status_message=str(e))
        raise e
    finally:
        if scheduler is not None:
            scheduler.stop()
    return


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

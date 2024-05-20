import argparse
import datetime
import logging
from typing import Callable, Optional

from clearml import Task

from ..utils.canceled_error import CanceledError
from ..utils.progress_status import ProgressStatus
from .clearml_shared_file_service import ClearMLSharedFileService
from .config import SETTINGS
from .smt_engine_build_job import SmtEngineBuildJob

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(str(__package__) + ".build_smt_engine")


def run(args: dict) -> None:
    progress: Optional[Callable[[ProgressStatus], None]] = None
    check_canceled: Optional[Callable[[], None]] = None
    task = None
    if args["clearml"]:
        task = Task.init()

        def clearml_check_canceled() -> None:
            if task.get_status() == "stopped":
                raise CanceledError

        check_canceled = clearml_check_canceled

        # get current time
        last_update_time = datetime.datetime.now()
        last_message = ""

        def clearml_progress(status: ProgressStatus) -> None:
            nonlocal last_update_time  # Add this line to access the outer variable
            nonlocal last_message  # Add this line to access the outer variable
            if status.percent_completed is not None:
                if status.message != last_message or (datetime.datetime.now() - last_update_time).seconds > 1:
                    last_update_time = datetime.datetime.now()
                    last_message = status.message
                    task.get_logger().report_single_value(name="progress", value=round(status.percent_completed, 4))
                    task.get_logger().report_text(f"Step: {status.step} Message: {status.message}")

        progress = clearml_progress
    else:

        # get current time
        last_update_time = datetime.datetime.now()
        last_message = ""

        def local_progress(status: ProgressStatus) -> None:
            nonlocal last_update_time  # Add this line to access the outer variable
            nonlocal last_message  # Add this line to access the outer variable
            if status.message != last_message or (datetime.datetime.now() - last_update_time).seconds > 1:
                last_update_time = datetime.datetime.now()
                last_message = status.message
                logger.info(f"Step: {status.step} Message: {status.message}")

        progress = local_progress

    try:
        logger.info("SMT Engine Build Job started")

        SETTINGS.update(args)
        shared_file_service = ClearMLSharedFileService(SETTINGS)
        smt_engine_build_job = SmtEngineBuildJob(SETTINGS, shared_file_service)
        smt_engine_build_job.run(progress=progress, check_canceled=check_canceled)
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
    parser.add_argument("--save-model", required=True, type=str, help="Save the model using the specified base name")
    parser.add_argument("--clearml", default=False, action="store_true", help="Initializes a ClearML task")
    parser.add_argument("--build-options", default=None, type=str, help="Build configurations")
    args = parser.parse_args()

    input_args = {k: v for k, v in vars(args).items() if v is not None}

    run(input_args)


if __name__ == "__main__":
    main()

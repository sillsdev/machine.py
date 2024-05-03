import argparse
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

logger = logging.getLogger(__package__ + ".build_smt_engine")


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

        def clearml_progress(status: ProgressStatus) -> None:
            if status.percent_completed is not None:
                task.get_logger().report_single_value(name="progress", value=round(status.percent_completed, 4))

        progress = clearml_progress

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

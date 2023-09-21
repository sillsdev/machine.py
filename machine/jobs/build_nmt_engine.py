import argparse
import logging
import os
from typing import Callable, Optional, cast

from clearml import Task

from ..utils.canceled_error import CanceledError
from .clearml_shared_file_service import ClearMLSharedFileService
from .config import SETTINGS
from .nmt_engine_build_job import NmtEngineBuildJob
from .nmt_model_factory import NmtModelFactory

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__package__ + ".build_nmt_engine")


def run(args: dict) -> None:
    check_canceled: Optional[Callable[[], None]] = None
    task = None
    if args["clearml"]:
        task = Task.init()

        def clearml_check_canceled() -> None:
            if task.get_status() in {"stopped", "stopping"}:
                raise CanceledError

        check_canceled = clearml_check_canceled

    try:
        logger.info("NMT Engine Build Job started")

        SETTINGS.update(args)
        SETTINGS.data_dir = os.path.expanduser(cast(str, SETTINGS.data_dir))

        logger.info(f"Config: {SETTINGS.as_dict()}")

        shared_file_service = ClearMLSharedFileService(SETTINGS)
        model_type = cast(str, SETTINGS.model_type).lower()
        nmt_model_factory: NmtModelFactory
        if model_type == "huggingface":
            from .huggingface.hugging_face_nmt_model_factory import HuggingFaceNmtModelFactory

            nmt_model_factory = HuggingFaceNmtModelFactory(SETTINGS, shared_file_service)
        else:
            raise RuntimeError("The model type is invalid.")

        job = NmtEngineBuildJob(SETTINGS, nmt_model_factory, shared_file_service)
        job.run(check_canceled)
        logger.info("Finished")
    except Exception as e:
        logger.exception(e, stack_info=True)
        if task:
            task.mark_failed(status_reason=type(e).__name__, status_message=str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--model-type", required=True, type=str, help="Model type")
    parser.add_argument("--engine-id", required=True, type=str, help="Engine id")
    parser.add_argument("--build-id", required=True, type=str, help="Build id")
    parser.add_argument("--src-lang", required=True, type=str, help="Source language tag")
    parser.add_argument("--trg-lang", required=True, type=str, help="Target language tag")
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps")
    parser.add_argument("--clearml", default=False, action="store_true", help="Initializes a ClearML task")
    args = parser.parse_args()

    run({k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main()

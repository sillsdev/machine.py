import argparse
import json
import logging
import os
from datetime import datetime
from typing import Callable, Optional, cast

import aiohttp
from clearml import Task

from ..utils.canceled_error import CanceledError
from ..utils.progress_status import ProgressStatus
from .async_scheduler import AsyncScheduler
from .clearml_shared_file_service import ClearMLSharedFileService
from .config import SETTINGS
from .smt_engine_build_job import SmtEngineBuildJob
from .smt_model_factory import SmtModelFactory

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(str(__package__) + ".build_smt_engine")


async def update_runtime_properties(task_id: str, base_url: str, token: str, runtime_props: dict) -> None:
    async with aiohttp.ClientSession(base_url=base_url, headers={"Authorization": f"Bearer {token}"}) as session:
        json = {"task": task_id, "runtime": runtime_props, "force": True}
        async with session.post("/tasks.edit", json=json) as response:
            response.raise_for_status()


def create_runtime_properties(task, percent_completed: Optional[int], message: Optional[str]) -> dict:
    runtime_props = task.data.runtime.copy() or {}
    if percent_completed is not None:
        runtime_props["progress"] = str(percent_completed)
    else:
        del runtime_props["progress"]
    if message is not None:
        runtime_props["message"] = message
    else:
        del runtime_props["message"]
    return runtime_props


def run(args: dict) -> None:
    progress: Callable[[ProgressStatus], None]
    check_canceled: Optional[Callable[[], None]] = None
    task = None
    last_percent_completed: Optional[int] = None
    last_message: Optional[str] = None
    scheduler: Optional[AsyncScheduler] = None
    if args["clearml"]:
        task = Task.init()

        scheduler = AsyncScheduler()

        last_check_canceled_time: Optional[datetime] = None

        def clearml_check_canceled() -> None:
            nonlocal last_check_canceled_time
            current_time = datetime.now()
            if last_check_canceled_time is None or (current_time - last_check_canceled_time).seconds > 20:
                if task.get_status() == "stopped":
                    raise CanceledError
                last_check_canceled_time = current_time

        check_canceled = clearml_check_canceled

        task.reload()

        last_progress_time: Optional[datetime] = None

        def clearml_progress(progress_status: ProgressStatus) -> None:
            nonlocal last_percent_completed
            nonlocal last_message
            nonlocal last_progress_time
            percent_completed: Optional[int] = None
            if progress_status.percent_completed is not None:
                percent_completed = round(progress_status.percent_completed * 100)
            message = progress_status.message
            if percent_completed != last_percent_completed or message != last_message:
                logger.info(f"{percent_completed}% - {message}")
                current_time = datetime.now()
                if last_progress_time is None or (current_time - last_progress_time).seconds > 1:
                    new_runtime_props = task.data.runtime.copy() or {}
                    new_runtime_props["progress"] = str(percent_completed)
                    new_runtime_props["message"] = message
                    scheduler.schedule(
                        update_runtime_properties(
                            task.id,
                            task.session.host,
                            task.session.token,
                            create_runtime_properties(task, percent_completed, message),
                        )
                    )
                    last_progress_time = current_time
                last_percent_completed = percent_completed
                last_message = message

        progress = clearml_progress
    else:

        def local_progress(progress_status: ProgressStatus) -> None:
            nonlocal last_percent_completed
            nonlocal last_message
            percent_completed: Optional[int] = None
            if progress_status.percent_completed is not None:
                percent_completed = round(progress_status.percent_completed * 100)
            message = progress_status.message
            if percent_completed != last_percent_completed or message != last_message:
                logger.info(f"{percent_completed}% - {message}")
                last_percent_completed = percent_completed
                last_message = message

        progress = local_progress

    try:
        logger.info("SMT Engine Build Job started")

        SETTINGS.update(args)
        model_type = cast(str, SETTINGS.model_type).lower()
        if "build_options" in SETTINGS:
            try:
                build_options = json.loads(cast(str, SETTINGS.build_options))
            except ValueError as e:
                raise ValueError("Build options could not be parsed: Invalid JSON") from e
            except TypeError as e:
                raise TypeError(f"Build options could not be parsed: {e}") from e
            SETTINGS.update({model_type: build_options})
        SETTINGS.data_dir = os.path.expanduser(cast(str, SETTINGS.data_dir))

        logger.info(f"Config: {SETTINGS.as_dict()}")

        shared_file_service = ClearMLSharedFileService(SETTINGS)
        smt_model_factory: SmtModelFactory
        if model_type == "thot":
            from .thot.thot_smt_model_factory import ThotSmtModelFactory

            smt_model_factory = ThotSmtModelFactory(SETTINGS)
        else:
            raise RuntimeError("The model type is invalid.")

        smt_engine_build_job = SmtEngineBuildJob(SETTINGS, smt_model_factory, shared_file_service)
        train_corpus_size, confidence = smt_engine_build_job.run(progress, check_canceled)
        if scheduler is not None and task is not None:
            scheduler.schedule(
                update_runtime_properties(
                    task.id, task.session.host, task.session.token, create_runtime_properties(task, 100, "Completed")
                )
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
    finally:
        if scheduler is not None:
            scheduler.stop()


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

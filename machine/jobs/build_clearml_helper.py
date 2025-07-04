import json
import logging
import os
from datetime import datetime
from typing import Callable, Optional, Union, cast

import aiohttp
from clearml import Task
from dynaconf.base import Settings

from ..utils.canceled_error import CanceledError
from ..utils.phased_progress_reporter import PhaseProgressStatus
from ..utils.progress_status import ProgressStatus
from .async_scheduler import AsyncScheduler


class ProgressInfo:
    last_percent_completed: Union[int, None] = 0
    last_message: Union[str, None] = ""
    last_progress_time: Union[datetime, None] = None
    last_check_canceled_time: Union[datetime, None] = None


def get_clearml_check_canceled(progress_info: ProgressInfo, task: Task) -> Callable[[], None]:

    def clearml_check_canceled() -> None:
        current_time = datetime.now()
        if (
            progress_info.last_check_canceled_time is None
            or (current_time - progress_info.last_check_canceled_time).seconds > 20
        ):
            if task.get_status() == "stopped":
                raise CanceledError
            progress_info.last_check_canceled_time = current_time

    return clearml_check_canceled


def get_clearml_progress_caller(
    progress_info: ProgressInfo, task: Task, scheduler: AsyncScheduler, logger: logging.Logger
) -> Callable[[ProgressStatus], None]:
    def clearml_progress(progress_status: ProgressStatus) -> None:
        percent_completed: Optional[int] = None
        if progress_status.percent_completed is not None:
            percent_completed = round(progress_status.percent_completed * 100)
        message = progress_status.message
        if percent_completed != progress_info.last_percent_completed or message != progress_info.last_message:
            logger.info(f"{percent_completed}% - {message}")
            current_time = datetime.now()
            if (
                progress_info.last_progress_time is None
                or (current_time - progress_info.last_progress_time).seconds > 1
            ):
                scheduler.schedule(
                    update_runtime_properties(
                        task,
                        create_runtime_properties(task, percent_completed, message, progress_status),
                    )
                )
                progress_info.last_progress_time = current_time
            progress_info.last_percent_completed = percent_completed
            progress_info.last_message = message

    return clearml_progress


def get_local_progress_caller(progress_info: ProgressInfo, logger: logging.Logger) -> Callable[[ProgressStatus], None]:

    def local_progress(progress_status: ProgressStatus) -> None:
        percent_completed: Optional[int] = None
        if progress_status.percent_completed is not None:
            percent_completed = round(progress_status.percent_completed * 100)
        message = progress_status.message
        if percent_completed != progress_info.last_percent_completed or message != progress_info.last_message:
            logger.info(f"{percent_completed}% - {message}")
            progress_info.last_percent_completed = percent_completed
            progress_info.last_message = message

    return local_progress


def update_settings(settings: Settings, args: dict):
    settings.update(args)
    settings.model_type = cast(str, settings.model_type).lower()
    if "build_options" in settings:
        try:
            build_options = json.loads(cast(str, settings.build_options))
        except ValueError as e:
            raise ValueError("Build options could not be parsed: Invalid JSON") from e
        except TypeError as e:
            raise TypeError(f"Build options could not be parsed: {e}") from e
        settings.update({settings.model_type: build_options})
    settings.data_dir = os.path.expanduser(cast(str, settings.data_dir))


async def update_runtime_properties(task, runtime_props: dict) -> None:
    current_runtime_properties = task.data.runtime or {}
    current_runtime_properties.update(runtime_props)
    async with aiohttp.ClientSession(
        base_url=task.session.host, headers={"Authorization": f"Bearer {task.session.token}"}
    ) as session:
        json = {"task": task.id, "runtime": runtime_props, "force": True}
        async with session.post("/tasks.edit", json=json) as response:
            response.raise_for_status()


def create_runtime_properties(
    task, percent_completed: Optional[int], message: Optional[str], status: Optional[ProgressStatus]
) -> dict:
    runtime_props = task.data.runtime.copy() or {}
    if percent_completed is not None:
        runtime_props["progress"] = str(percent_completed)
    if message is not None:
        runtime_props["message"] = message
    # Report the step within the phase
    if status is not None and isinstance(status, PhaseProgressStatus):
        if status.phase_stage is not None:
            if status.phase_step is not None:
                runtime_props[f"{status.phase_stage}_step"] = str(status.phase_step)
            if status.step_count is not None:
                runtime_props[f"{status.phase_stage}_step_count"] = str(status.step_count)
    return runtime_props

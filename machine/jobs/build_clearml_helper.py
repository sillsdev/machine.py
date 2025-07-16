import json
import logging
import os
from datetime import datetime
from typing import Callable, Optional, Union, cast

from clearml import Task
from dynaconf.base import Settings

from ..utils.canceled_error import CanceledError
from ..utils.phased_progress_reporter import PhaseProgressStatus
from ..utils.progress_status import ProgressStatus


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
    progress_info: ProgressInfo, task: Task, logger: logging.Logger
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
                report_clearml_progress(
                    task=task, percent_completed=percent_completed, message=message, progress_status=progress_status
                )
                progress_info.last_progress_time = current_time
            progress_info.last_percent_completed = percent_completed
            progress_info.last_message = message

    return clearml_progress


def report_clearml_progress(
    task: Task,
    percent_completed: Optional[int] = None,
    message: Optional[str] = None,
    progress_status: Optional[ProgressStatus] = None,
) -> None:
    if percent_completed is not None:
        task.set_progress(percent_completed)
    props = []
    if message is not None:
        props.append({"type": str, "name": "message", "description": "Build Message", "value": message})
    # Report the step within the phase
    if progress_status is not None and isinstance(progress_status, PhaseProgressStatus):
        if progress_status.phase_stage is not None:
            if progress_status.phase_step is not None:
                props.append(
                    {
                        "type": int,
                        "name": f"{progress_status.phase_stage}_step",
                        "description": "Phase Step",
                        "value": progress_status.phase_step,
                    }
                )
            if progress_status.step_count is not None:
                props.append(
                    {
                        "type": int,
                        "name": f"{progress_status.phase_stage}_step_count",
                        "description": "Maximum Phase Step",
                        "value": progress_status.step_count,
                    }
                )
    if len(props) > 0:
        task.set_user_properties(*props)


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


def update_settings(settings: Settings, args: dict, task: Optional[Task], logger: logging.Logger):
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
        if "align_pretranslations" in build_options:
            settings.update({"align_pretranslations": build_options["align_pretranslations"]})
        if task is not None and "tags" in build_options:
            tags = build_options["tags"]
            if isinstance(tags, str) or (isinstance(tags, list) and all(isinstance(tag, str) for tag in tags)):
                task.add_tags(tags)
    settings.data_dir = os.path.expanduser(cast(str, settings.data_dir))
    logger.info(f"Config: {settings.as_dict()}")

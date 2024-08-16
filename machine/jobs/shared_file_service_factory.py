from enum import IntEnum, auto
from typing import Any, Union

from .clearml_shared_file_service import ClearMLSharedFileService
from .local_shared_file_service import LocalSharedFileService
from .shared_file_service_base import SharedFileServiceBase


class SharedFileServiceType(IntEnum):
    LOCAL = auto()
    CLEARML = auto()


def get_shared_file_service(type: Union[str, SharedFileServiceType], config: Any) -> SharedFileServiceBase:
    if isinstance(type, str):
        type = SharedFileServiceType[type.upper()]
    if type == SharedFileServiceType.LOCAL:
        return LocalSharedFileService(config)
    elif type == SharedFileServiceType.CLEARML:
        return ClearMLSharedFileService(config)

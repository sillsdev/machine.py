import logging
import time
from pathlib import Path
from typing import Callable

from clearml import StorageManager

from .shared_file_service_base import SharedFileServiceBase

logger = logging.getLogger(__name__)


class ClearMLSharedFileService(SharedFileServiceBase):
    def download_file(self, path: str) -> Path:
        local_folder = str(self._data_dir)
        file_path = try_n_times(
            lambda: StorageManager.download_file(self._get_uri(path), local_folder, skip_zero_size_check=True)
        )
        if file_path is None:
            raise RuntimeError(f"Failed to download file: {self._get_uri(path)}")
        return Path(file_path)

    def _download_folder(self, path: str) -> Path:
        local_folder = str(self._data_dir)
        folder_path = try_n_times(lambda: StorageManager.download_folder(self._get_uri(path), local_folder))
        if folder_path is None:
            raise RuntimeError(f"Failed to download folder: {self._get_uri(path)}")
        return Path(folder_path) / path

    def _exists_file(self, path: str) -> bool:
        return try_n_times(lambda: StorageManager.exists_file(self._get_uri(path)))  # type: ignore

    def _upload_file(self, path: str, local_file_path: Path) -> None:
        final_destination = try_n_times(lambda: StorageManager.upload_file(str(local_file_path), self._get_uri(path)))
        if final_destination is None:
            logger.error((f"Failed to upload file {str(local_file_path)} " f"to {self._get_uri(path)}."))

    def _upload_folder(self, path: str, local_folder_path: Path) -> None:
        final_destination = try_n_times(
            lambda: StorageManager.upload_folder(str(local_folder_path), self._get_uri(path))
        )
        if final_destination is None:
            logger.error((f"Failed to upload folder {str(local_folder_path)} " f"to {self._get_uri(path)}."))

    def _get_uri(self, path: str) -> str:
        return f"{self._shared_file_uri}/{self._shared_file_folder}/{path}"


def try_n_times(func: Callable, n=10):
    for i in range(n):
        try:
            return func()
        except Exception as e:
            if i < n - 1:
                logger.exception(f"Failed {i+1} of {n} times.  Retrying.")
                time.sleep(5)
            else:
                raise e

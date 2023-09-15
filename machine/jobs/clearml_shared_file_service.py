import logging
import time
from pathlib import Path
from typing import Callable, Optional

from clearml import StorageManager

from .shared_file_service import SharedFileService

logger = logging.getLogger(__name__)


class ClearMLSharedFileService(SharedFileService):
    def _download_file(self, path: str, cache: bool = False) -> Path:
        uri = f"{self._shared_file_uri}/{path}"
        local_folder: Optional[str] = None
        if not cache:
            local_folder = str(self._data_dir)
        file_path = try_n_times(lambda: StorageManager.download_file(uri, local_folder))
        if file_path is None:
            raise RuntimeError(f"Failed to download file: {uri}")
        return Path(file_path)

    def _download_folder(self, path: str, cache: bool = False) -> Path:
        uri = f"{self._shared_file_uri}/{path}"
        local_folder: Optional[str] = None
        if not cache:
            local_folder = str(self._data_dir)
        folder_path = try_n_times(lambda: StorageManager.download_folder(uri, local_folder))
        if folder_path is None:
            raise RuntimeError(f"Failed to download folder: {uri}")
        return Path(folder_path) / path

    def _upload_file(self, path: str, local_file_path: Path) -> None:
        final_destination = try_n_times(
            lambda: StorageManager.upload_file(str(local_file_path), f"{self._shared_file_uri}/{path}")
        )
        if final_destination is None:
            logger.error(f"Failed to upload file {str(local_file_path)} to {self._shared_file_uri}/{path}.")

    def _upload_folder(self, path: str, local_folder_path: Path) -> None:
        final_destination = try_n_times(
            lambda: StorageManager.upload_folder(str(local_folder_path), f"{self._shared_file_uri}/{path}")
        )
        if final_destination is None:
            logger.error(f"Failed to upload folder {str(local_folder_path)} to {self._shared_file_uri}/{path}.")


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

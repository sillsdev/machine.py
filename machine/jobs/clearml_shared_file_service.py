from pathlib import Path
from typing import Optional

from clearml import StorageManager

from .shared_file_service import SharedFileService


class ClearMLSharedFileService(SharedFileService):
    def _download_file(self, path: str, cache: bool = False) -> Path:
        uri = f"{self._shared_file_uri}/{path}"
        local_folder: Optional[str] = None
        if not cache:
            local_folder = str(self._data_dir)
        file_path = StorageManager.download_file(uri, local_folder)
        if file_path is None:
            raise RuntimeError(f"Failed to download file: {uri}")
        return Path(file_path)

    def _download_folder(self, path: str, cache: bool = False) -> Path:
        uri = f"{self._shared_file_uri}/{path}"
        local_folder: Optional[str] = None
        if not cache:
            local_folder = str(self._data_dir)
        folder_path = StorageManager.download_folder(uri, local_folder)
        if folder_path is None:
            raise RuntimeError(f"Failed to download folder: {uri}")
        return Path(folder_path) / path

    def _upload_file(self, path: str, local_file_path: Path) -> None:
        StorageManager.upload_file(str(local_file_path), f"{self._shared_file_uri}/{path}")

    def _upload_folder(self, path: str, local_folder_path: Path) -> None:
        StorageManager.upload_folder(str(local_folder_path), f"{self._shared_file_uri}/{path}")

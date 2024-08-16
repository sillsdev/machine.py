import logging
import shutil
from pathlib import Path

from .shared_file_service_base import SharedFileServiceBase

logger = logging.getLogger(__name__)


class LocalSharedFileService(SharedFileServiceBase):
    def download_file(self, path: str) -> Path:
        src_path = self._get_path(path)
        dst_path = self._data_dir / self._shared_file_folder / path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst_path)
        return dst_path

    def _download_folder(self, path: str) -> Path:
        src_path = self._get_path(path)
        dst_path = self._data_dir / self._shared_file_folder / path
        dst_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst_path)
        return dst_path

    def _exists_file(self, path: str) -> bool:
        return self._get_path(path).exists()

    def _upload_file(self, path: str, local_file_path: Path) -> None:
        dst_path = self._get_path(path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_file_path, dst_path)

    def _upload_folder(self, path: str, local_folder_path: Path) -> None:
        dst_path = self._get_path(path)
        dst_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_folder_path, dst_path)

    def _get_path(self, name: str) -> Path:
        # Don't use shared file folder for local files
        return Path(f"{self._shared_file_uri}/{name}")

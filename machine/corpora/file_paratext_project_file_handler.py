import os
from pathlib import Path
from typing import BinaryIO, Optional

from ..utils.typeshed import StrPath
from .paratext_project_file_handler import ParatextProjectFileHandler
from .usfm_stylesheet import UsfmStylesheet


class FileParatextProjectFileHandler(ParatextProjectFileHandler):
    def __init__(self, project_dir: StrPath) -> None:
        self._project_dir = Path(project_dir)

    def exists(self, file_name: str) -> bool:
        for actual_file_name in os.listdir(self._project_dir):
            if actual_file_name.lower() == file_name.lower():
                return True
        return False

    def open(self, file_name: str) -> BinaryIO:
        for actual_file_name in os.listdir(self._project_dir):
            if actual_file_name.lower() == file_name.lower():
                return open(self._project_dir / actual_file_name, "rb")
        return open(self._project_dir / file_name, "rb")

    def find(self, extension: str) -> Optional[Path]:
        return next(self._project_dir.glob(f"*{extension}"), None)

    def create_stylesheet(self, file_name: str) -> UsfmStylesheet:
        custom_stylesheet_file_name = "custom.sty"
        for actual_file_name in os.listdir(self._project_dir):
            if actual_file_name.lower() == custom_stylesheet_file_name:
                custom_stylesheet_file_name = actual_file_name
                break
        custom_stylesheet_path = self._project_dir / custom_stylesheet_file_name
        return UsfmStylesheet(
            file_name,
            custom_stylesheet_path if custom_stylesheet_path.is_file() else None,
        )

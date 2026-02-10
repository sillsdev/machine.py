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
        return self._get_file_name(file_name) is not None

    def open(self, file_name: str) -> BinaryIO:
        actual_file_name = self._get_file_name(file_name)
        if actual_file_name is not None:
            file_name = actual_file_name
        return open(self._project_dir / file_name, "rb")

    def find(self, extension: str) -> Optional[Path]:
        return next(self._project_dir.glob(f"*{extension}"), None)

    def create_stylesheet(self, file_name: str) -> UsfmStylesheet:
        custom_stylesheet_file_name = self._get_file_name("custom.sty")
        if custom_stylesheet_file_name is None:
            custom_stylesheet_file_name = "custom.sty"
        custom_stylesheet_path = self._project_dir / custom_stylesheet_file_name
        return UsfmStylesheet(
            file_name,
            custom_stylesheet_path if custom_stylesheet_path.is_file() else None,
        )

    def _get_file_name(self, case_insensitive_file_name: str) -> Optional[str]:
        for actual_file_name in os.listdir(self._project_dir):
            if actual_file_name.lower() == case_insensitive_file_name.lower():
                return actual_file_name
        return None

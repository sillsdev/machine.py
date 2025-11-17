from pathlib import Path
from typing import BinaryIO, Optional

from ..utils.typeshed import StrPath
from .paratext_project_file_handler import ParatextProjectFileHandler
from .usfm_stylesheet import UsfmStylesheet


class FileParatextProjectFileHandler(ParatextProjectFileHandler):
    def __init__(self, project_dir: StrPath) -> None:
        self._project_dir = Path(project_dir)

    def exists(self, file_name: str) -> bool:
        return (self._project_dir / file_name).is_file()

    def open(self, file_name: str) -> BinaryIO:
        return open(self._project_dir / file_name, "rb")

    def find(self, extension: str) -> Optional[Path]:
        return next(self._project_dir.glob(f"*{extension}"), None)

    def create_stylesheet(self, file_name: str) -> UsfmStylesheet:
        custom_stylesheet_filename = self._project_dir / "custom.sty"
        return UsfmStylesheet(
            file_name,
            custom_stylesheet_filename if custom_stylesheet_filename.is_file() else None,
        )

from pathlib import Path
from typing import BinaryIO, Optional

from ..utils.typeshed import StrPath
from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .usfm_stylesheet import UsfmStylesheet


class FileParatextProjectSettingsParser(ParatextProjectSettingsParserBase):
    def __init__(self, project_dir: StrPath) -> None:
        self._project_dir = Path(project_dir)

    def _create_stylesheet(self, file_name: StrPath) -> UsfmStylesheet:
        custom_stylesheet_filename = self._project_dir / file_name
        return UsfmStylesheet(
            file_name,
            custom_stylesheet_filename if custom_stylesheet_filename.is_file() else None,
        )

    def _exists(self, file_name: StrPath) -> bool:
        return (self._project_dir / file_name).is_file()

    def _find(self, extension: str) -> Optional[Path]:
        return next(self._project_dir.glob(f"*{extension}"), None)

    def _open(self, file_name: StrPath) -> BinaryIO:
        return open(self._project_dir / file_name, "rb")

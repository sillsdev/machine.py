import os
from tempfile import mkstemp
from typing import Optional

from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .usfm_stylesheet import UsfmStylesheet


class ZipParatextProjectSettingsParserBase(ParatextProjectSettingsParserBase):
    def _create_stylesheet(self, file_name: str) -> UsfmStylesheet:
        stylesheet_temp_path: Optional[str] = None
        stylesheet_temp_fd: Optional[int] = None
        custom_stylesheet_temp_path: Optional[str] = None
        custom_stylesheet_temp_fd: Optional[int] = None
        try:
            stylesheet_path: str = file_name
            if self._exists(file_name):
                stylesheet_temp_fd, stylesheet_temp_path = mkstemp()
                with (
                    self._open(file_name) as source,
                    open(stylesheet_temp_fd, "wb", closefd=False) as stylesheet_temp_file,
                ):
                    stylesheet_temp_file.write(source.read())
                stylesheet_path = stylesheet_temp_path
            custom_stylesheet_path: Optional[str] = None
            if self._exists("custom.sty"):
                custom_stylesheet_temp_fd, custom_stylesheet_temp_path = mkstemp()
                with (
                    self._open("custom.sty") as source,
                    open(custom_stylesheet_temp_fd, "wb", closefd=False) as custom_stylesheet_temp_file,
                ):
                    custom_stylesheet_temp_file.write(source.read())
                custom_stylesheet_path = custom_stylesheet_temp_path
            return UsfmStylesheet(stylesheet_path, custom_stylesheet_path)
        finally:
            if stylesheet_temp_fd is not None:
                os.close(stylesheet_temp_fd)
            if stylesheet_temp_path is not None:
                os.remove(stylesheet_temp_path)
            if custom_stylesheet_temp_fd is not None:
                os.close(custom_stylesheet_temp_fd)
            if custom_stylesheet_temp_path is not None:
                os.remove(custom_stylesheet_temp_path)

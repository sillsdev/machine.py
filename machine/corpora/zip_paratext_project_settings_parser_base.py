from tempfile import TemporaryFile
from typing import Optional

from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .usfm_stylesheet import UsfmStylesheet


class ZipParatextProjectSettingsParserBase(ParatextProjectSettingsParserBase):
    def _create_stylesheet(self, file_name: str) -> UsfmStylesheet:
        with TemporaryFile() as stylesheet_temp_file, TemporaryFile() as custom_stylesheet_temp_file:
            stylesheet_path: str = file_name
            if self._exists(file_name):
                with self._open(file_name) as source:
                    stylesheet_temp_file.write(source.read())
                stylesheet_path = stylesheet_temp_file.name
            custom_stylesheet_path: Optional[str] = None
            if self._exists("custom.sty"):
                with self._open("custom.sty") as source:
                    custom_stylesheet_temp_file.write(source.read())
                custom_stylesheet_path = custom_stylesheet_temp_file.name
            return UsfmStylesheet(stylesheet_path, custom_stylesheet_path)

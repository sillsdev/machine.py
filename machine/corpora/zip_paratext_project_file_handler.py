import os
from io import BytesIO
from tempfile import mkstemp
from typing import BinaryIO, Optional, cast
from zipfile import ZipFile

from .paratext_project_file_handler import ParatextProjectFileHandler
from .usfm_stylesheet import UsfmStylesheet


class ZipParatextProjectFileHandler(ParatextProjectFileHandler):
    def __init__(self, archive: ZipFile) -> None:
        self._archive = archive

    def exists(self, file_name: str) -> bool:
        return file_name in self._archive.namelist()

    def find(self, extension: str) -> Optional[str]:
        for entry in self._archive.namelist():
            if entry.endswith(extension):
                return entry
        return None

    def open(self, file_name: str) -> Optional[BinaryIO]:
        if file_name in self._archive.namelist():
            return BytesIO(self._archive.read(file_name))
        return None

    def create_stylesheet(self, file_name: str) -> UsfmStylesheet:
        stylesheet_temp_path: Optional[str] = None
        stylesheet_temp_fd: Optional[int] = None
        custom_stylesheet_temp_path: Optional[str] = None
        custom_stylesheet_temp_fd: Optional[int] = None
        try:
            stylesheet_path: str = file_name
            if self.exists(file_name):
                stylesheet_temp_fd, stylesheet_temp_path = mkstemp()
                with (
                    cast(BinaryIO, self.open(file_name)) as source,
                    open(stylesheet_temp_fd, "wb", closefd=False) as stylesheet_temp_file,
                ):
                    stylesheet_temp_file.write(source.read())
                stylesheet_path = stylesheet_temp_path
            custom_stylesheet_path: Optional[str] = None
            if self.exists("custom.sty"):
                custom_stylesheet_temp_fd, custom_stylesheet_temp_path = mkstemp()
                with (
                    cast(BinaryIO, self.open("custom.sty")) as source,
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

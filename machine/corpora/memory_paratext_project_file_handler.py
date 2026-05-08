from io import BytesIO
from typing import BinaryIO, Dict, Optional

from .paratext_project_file_handler import ParatextProjectFileHandler
from .usfm_stylesheet import UsfmStylesheet


class MemoryParatextProjectFileHandler(ParatextProjectFileHandler):
    def __init__(self, files: Dict[str, str]) -> None:
        self.files = files

    def exists(self, file_name: str) -> bool:
        return file_name in self.files

    def open(self, file_name: str) -> BinaryIO:
        return BytesIO(self.files[file_name].encode("utf-8"))

    def find(self, extension: str) -> Optional[str]:
        for name in self.files:
            if name.endswith(extension):
                return name
        return None

    def create_stylesheet(self, file_name: str) -> UsfmStylesheet:
        return UsfmStylesheet(file_name)

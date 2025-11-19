from abc import ABC, abstractmethod
from typing import BinaryIO

from .usfm_stylesheet import UsfmStylesheet


class ParatextProjectFileHandler(ABC):
    @abstractmethod
    def exists(self, file_name: str) -> bool: ...

    @abstractmethod
    def open(self, file_name: str) -> BinaryIO: ...

    @abstractmethod
    def find(self, extension: str) -> str: ...

    @abstractmethod
    def create_stylesheet(self, file_name: str) -> UsfmStylesheet: ...

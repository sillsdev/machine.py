from abc import ABC, abstractmethod
from typing import BinaryIO


class StreamContainer(ABC):
    @abstractmethod
    def __enter__(self) -> "StreamContainer":
        ...

    @abstractmethod
    def __exit__(self, type, value, traceback) -> None:
        ...

    @abstractmethod
    def open_stream(self) -> BinaryIO:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

from abc import ABC, abstractmethod
from typing import Any, BinaryIO


class StreamContainer(ABC):
    @abstractmethod
    def __enter__(self) -> "StreamContainer":
        ...

    @abstractmethod
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        ...

    @abstractmethod
    def open_stream(self) -> BinaryIO:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

from io import RawIOBase
from typing import BinaryIO


class BoundedStream(RawIOBase):
    """A read-only stream that raises an `OSError` once more than `max_size` bytes have been read.

    Closing the stream closes the inner stream.
    """

    def __init__(self, inner_stream: BinaryIO, max_size: int) -> None:
        if max_size < 0:
            raise ValueError("max_size must not be negative.")
        self._inner_stream = inner_stream
        self._max_size = max_size
        self._total_bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        data = self._inner_stream.read(len(buffer))
        self._total_bytes_read += len(data)
        if self._total_bytes_read > self._max_size:
            raise OSError(f"Stream operation aborted. Exceeded maximum limit of {self._max_size} bytes.")
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        try:
            self._inner_stream.close()
        finally:
            super().close()

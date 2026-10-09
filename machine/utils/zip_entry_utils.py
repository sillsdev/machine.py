from typing import Union
from zipfile import BadZipFile, ZipFile, ZipInfo

from .bounded_stream import BoundedStream

DEFAULT_MAX_UNCOMPRESSED_SIZE = 100 * 1024 * 1024  # 100 MB
DEFAULT_MAX_COMPRESSION_RATIO = 100.0  # 100:1 ratio limit


def open_bounded_stream(
    archive: ZipFile,
    entry: Union[str, ZipInfo],
    max_uncompressed_size: int = DEFAULT_MAX_UNCOMPRESSED_SIZE,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
) -> BoundedStream:
    """Opens a zip entry for reading, refusing entries that are too large or too highly compressed.

    Raises `BadZipFile` if the entry's declared size or compression ratio exceeds the limits. Reading more than
    `max_uncompressed_size` bytes from the returned stream raises an `OSError`.
    """
    if isinstance(entry, str):
        entry = archive.getinfo(entry)

    if entry.file_size > max_uncompressed_size:
        raise BadZipFile("Entry uncompressed size exceeds maximum allowed limit.")

    if entry.compress_size > 0 and entry.file_size / entry.compress_size > max_compression_ratio:
        raise BadZipFile("Compression ratio exceeds safe threshold.")

    return BoundedStream(archive.open(entry, "r"), max_uncompressed_size)

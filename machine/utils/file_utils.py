from os import PathLike
from typing import IO, BinaryIO, cast

from charset_normalizer import from_fp, from_path

from .typeshed import StrPath


def detect_encoding(filename: StrPath) -> str:
    match = from_path(cast(PathLike, filename)).best()
    if match is None:
        return "utf-8"
    return match.encoding


def detect_encoding_from_stream(stream: IO[bytes]) -> str:
    match = from_fp(cast(BinaryIO, stream)).best()
    if match is None:
        return "utf-8"
    return match.encoding

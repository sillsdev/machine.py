from io import BytesIO
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from pytest import raises

from machine.utils.bounded_stream import BoundedStream
from machine.utils.zip_entry_utils import open_bounded_stream


def create_zip(name: str, content: bytes) -> ZipFile:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return ZipFile(buffer, "r")


def test_open_bounded_stream_returns_readable_stream() -> None:
    with create_zip("test.txt", b"Hello World") as archive:
        with open_bounded_stream(archive, "test.txt", max_uncompressed_size=100) as stream:
            assert stream.read() == b"Hello World"


def test_open_bounded_stream_empty_entry() -> None:
    with create_zip("empty.txt", b"") as archive:
        with open_bounded_stream(archive, "empty.txt") as stream:
            assert stream.read() == b""


def test_open_bounded_stream_header_size_exceeded() -> None:
    with create_zip("large.txt", bytes(200)) as archive:
        with raises(BadZipFile):
            open_bounded_stream(archive, "large.txt", max_uncompressed_size=100)


def test_open_bounded_stream_compression_ratio_exceeded() -> None:
    with create_zip("bomb.txt", bytes(10_000)) as archive:
        with raises(BadZipFile):
            open_bounded_stream(archive, "bomb.txt", max_uncompressed_size=20_000, max_compression_ratio=2.0)


def test_bounded_stream_runtime_expansion_exceeds_limit() -> None:
    with BoundedStream(BytesIO(b"1234567890"), max_size=5) as stream:
        assert stream.read(4) == b"1234"
        with raises(OSError):
            stream.read(4)

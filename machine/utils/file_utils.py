from typing import IO

from chardet import UniversalDetector

from .typeshed import StrPath


def detect_encoding(filename: StrPath) -> str:
    with open(filename, "rb") as file:
        return detect_encoding_from_stream(file)


def detect_encoding_from_stream(stream: IO[bytes]) -> str:
    detector = UniversalDetector()
    for line in stream:
        detector.feed(line)
        if detector.done:
            break
    return detector.close()["encoding"]

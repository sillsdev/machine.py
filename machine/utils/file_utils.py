from chardet import UniversalDetector

from .typeshed import StrPath


def detect_encoding(filename: StrPath) -> str:
    detector = UniversalDetector()
    with open(filename, "rb") as file:
        for line in file:
            detector.feed(line)
            if detector.done:
                break
    return detector.close()["encoding"]

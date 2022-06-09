import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, List, TextIO

import json_stream
from clearml import StorageManager

from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..utils.context_managed_generator import ContextManagedGenerator

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class PretranslationInfo(TypedDict):
    corpusId: str
    textId: str
    refs: List[str]
    segment: str


class PretranslationWriter:
    def __init__(self, file: TextIO) -> None:
        self._file = file
        self._first = True

    def write(self, pi: PretranslationInfo) -> None:
        if not self._first:
            self._file.write(",\n")
        self._file.write("    " + json.dumps(pi))
        self._first = False


class SharedFileService:
    def __init__(self, config: dict) -> None:
        self._config = config

    def init(self) -> None:
        self._data_dir.mkdir(exist_ok=True)

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(self._download_file("train.src.txt"))

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(self._download_file("train.trg.txt"))

    def get_source_pretranslations(self) -> ContextManagedGenerator[PretranslationInfo, None, None]:
        src_pretranslate_path = self._download_file("pretranslate.src.json")

        def generator() -> Generator[PretranslationInfo, None, None]:
            with src_pretranslate_path.open("r", encoding="utf-8-sig") as file:
                for pi in json_stream.load(file):
                    yield PretranslationInfo(
                        corpusId=pi["corpusId"], textId=pi["textId"], refs=list(pi["refs"]), segment=pi["segment"]
                    )

        return ContextManagedGenerator(generator())

    @contextmanager
    def open_target_pretranslation_writer(self) -> Iterator[PretranslationWriter]:
        target_pretranslate_path = self._data_dir / "pretranslate.trg.json"
        with target_pretranslate_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write("[\n")
            yield PretranslationWriter(file)
            file.write("\n]\n")
        self._upload_file("pretranslate.trg.json")

    @property
    def _data_dir(self) -> Path:
        return Path(self._config["data_dir"], self._config["build_id"])

    @property
    def _build_uri(self) -> str:
        build_uri: str = self._config["build_uri"]
        build_uri = build_uri.rstrip("/")
        return build_uri

    def _download_file(self, filename: str) -> Path:
        uri = f"{self._build_uri}/{filename}"
        file_path = StorageManager.download_file(uri, str(self._data_dir))
        if file_path is None:
            raise RuntimeError(f"Failed to download file: {uri}")
        return Path(file_path)

    def _upload_file(self, filename: str) -> None:
        file_path = self._data_dir / filename
        StorageManager.upload_file(str(file_path), f"{self._build_uri}/{filename}")

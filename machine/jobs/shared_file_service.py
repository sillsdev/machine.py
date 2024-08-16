import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, MutableMapping, TextIO, TypedDict

import json_stream

from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..utils.context_managed_generator import ContextManagedGenerator


class PretranslationInfo(TypedDict):
    corpusId: str  # noqa: N815
    textId: str  # noqa: N815
    refs: List[str]
    translation: str


class WordAlignmentInfo(TypedDict):
    refs: List[str]
    column_count: int
    row_count: int
    alignmnent: str


class DictToJsonWriter:
    def __init__(self, file: TextIO) -> None:
        self._file = file
        self._first = True

    # Use MutableMapping rather than TypeDict to allow for more flexible input
    def write(self, pi: MutableMapping) -> None:
        if not self._first:
            self._file.write(",\n")
        self._file.write("    " + json.dumps(pi))
        self._first = False


class SharedFileService(ABC):
    def __init__(
        self,
        config: Any,
        source_filename: str = "train.src.txt",
        target_filename: str = "train.trg.txt",
    ) -> None:
        self._config = config
        self._source_filename = source_filename
        self._target_filename = target_filename

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(self._download_file(f"{self._build_path}/{self._source_filename}"))

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(self._download_file(f"{self._build_path}/{self._target_filename}"))

    def exists_source_corpus(self) -> bool:
        return self._exists_file(f"{self._build_path}/{self._source_filename}")

    def exists_target_corpus(self) -> bool:
        return self._exists_file(f"{self._build_path}/{self._target_filename}")

    def get_source_pretranslations(self) -> ContextManagedGenerator[PretranslationInfo, None, None]:
        src_pretranslate_path = self._download_file(f"{self._build_path}/pretranslate.src.json")

        def generator() -> Generator[PretranslationInfo, None, None]:
            with src_pretranslate_path.open("r", encoding="utf-8-sig") as file:
                for pi in json_stream.load(file):
                    yield PretranslationInfo(
                        corpusId=pi["corpusId"],
                        textId=pi["textId"],
                        refs=list(pi["refs"]),
                        translation=pi["translation"],
                    )

        return ContextManagedGenerator(generator())

    @contextmanager
    def open_target_pretranslation_writer(self) -> Iterator[DictToJsonWriter]:
        return self._open_target_writer("pretranslate.trg.json")

    @contextmanager
    def open_target_alignment_writer(self) -> Iterator[DictToJsonWriter]:
        return self._open_target_writer("word_alignments.json")

    def _open_target_writer(self, filename) -> Iterator[DictToJsonWriter]:
        build_dir = self._data_dir / self._shared_file_folder / self._build_path
        build_dir.mkdir(parents=True, exist_ok=True)
        target_path = build_dir / filename
        with target_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write("[\n")
            yield DictToJsonWriter(file)
            file.write("\n]\n")
        self._upload_file(f"{self._build_path}/{filename}", target_path)

    def save_model(self, model_path: Path, destination: str) -> None:
        if model_path.is_file():
            self._upload_file(destination, model_path)
        else:
            self._upload_folder(destination, model_path)

    @property
    def _data_dir(self) -> Path:
        return Path(self._config.data_dir)

    @property
    def _build_path(self) -> str:
        return f"builds/{self._config.build_id}"

    @property
    def _engine_id(self) -> str:
        return self._config.engine_id

    @property
    def _shared_file_uri(self) -> str:
        shared_file_uri: str = self._config.shared_file_uri
        return shared_file_uri.rstrip("/")

    @property
    def _shared_file_folder(self) -> str:
        shared_file_folder: str = self._config.shared_file_folder
        return shared_file_folder.rstrip("/")

    @abstractmethod
    def _download_file(self, path: str) -> Path: ...

    @abstractmethod
    def _download_folder(self, path: str) -> Path: ...

    @abstractmethod
    def _exists_file(self, path: str) -> bool: ...

    @abstractmethod
    def _upload_file(self, path: str, local_file_path: Path) -> None: ...

    @abstractmethod
    def _upload_folder(self, path: str, local_folder_path: Path) -> None: ...

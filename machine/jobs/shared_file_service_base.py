import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, TextIO


class DictToJsonWriter:
    def __init__(self, file: TextIO) -> None:
        self._file = file
        self._first = True

    def write(self, pi: object) -> None:
        if not self._first:
            self._file.write(",\n")
        self._file.write("    " + json.dumps(pi))
        self._first = False


class SharedFileServiceBase(ABC):
    def __init__(
        self,
        config: Any,
    ) -> None:
        self._config = config

    def upload_path(self, path: Path, destination: str) -> None:
        if path.is_file():
            self._upload_file(destination, path)
        else:
            self._upload_folder(destination, path)

    def open_target_writer(self, filename) -> Iterator[DictToJsonWriter]:
        build_dir = self._data_dir / self._shared_file_folder / self.build_path
        build_dir.mkdir(parents=True, exist_ok=True)
        target_path = build_dir / filename
        with target_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write("[\n")
            yield DictToJsonWriter(file)
            file.write("\n]\n")
        self._upload_file(f"{self.build_path}/{filename}", target_path)

    @property
    def build_path(self) -> str:
        return f"builds/{self._config.build_id}"

    @property
    def _data_dir(self) -> Path:
        return Path(self._config.data_dir)

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
    def download_file(self, path: str) -> Path: ...

    @abstractmethod
    def _download_folder(self, path: str) -> Path: ...

    @abstractmethod
    def _exists_file(self, path: str) -> bool: ...

    @abstractmethod
    def _upload_file(self, path: str, local_file_path: Path) -> None: ...

    @abstractmethod
    def _upload_folder(self, path: str, local_folder_path: Path) -> None: ...

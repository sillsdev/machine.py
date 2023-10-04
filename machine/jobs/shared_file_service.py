import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, TextIO, TypedDict

import json_stream

from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..utils.context_managed_generator import ContextManagedGenerator


class PretranslationInfo(TypedDict):
    corpusId: str
    textId: str
    refs: List[str]
    translation: str


class PretranslationWriter:
    def __init__(self, file: TextIO) -> None:
        self._file = file
        self._first = True

    def write(self, pi: PretranslationInfo) -> None:
        if not self._first:
            self._file.write(",\n")
        self._file.write("    " + json.dumps(pi))
        self._first = False


class SharedFileService(ABC):
    def __init__(self, config: Any) -> None:
        self._config = config

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(self._download_file(f"builds/{self._build_id}/train.src.txt"))

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(self._download_file(f"builds/{self._build_id}/train.trg.txt"))

    def exists_source_corpus(self) -> bool:
        return self._exists_file(f"builds/{self._build_id}/train.src.txt")

    def exists_target_corpus(self) -> bool:
        return self._exists_file(f"builds/{self._build_id}/train.trg.txt")

    def get_source_pretranslations(self) -> ContextManagedGenerator[PretranslationInfo, None, None]:
        src_pretranslate_path = self._download_file(f"builds/{self._build_id}/pretranslate.src.json")

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
    def open_target_pretranslation_writer(self) -> Iterator[PretranslationWriter]:
        build_id: str = self._config.build_id
        build_dir = self._data_dir / "builds" / build_id
        build_dir.mkdir(parents=True, exist_ok=True)
        target_pretranslate_path = build_dir / "pretranslate.trg.json"
        with target_pretranslate_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write("[\n")
            yield PretranslationWriter(file)
            file.write("\n]\n")
        self._upload_file(f"builds/{self._build_id}/pretranslate.trg.json", target_pretranslate_path)

    def get_parent_model(self, language_tag: str) -> Path:
        return self._download_folder(f"parent_models/{language_tag}", cache=True)

    def save_model(self, model_dir: Path) -> None:
        self._upload_folder(f"models/{self._engine_id}", model_dir)

    @property
    def _data_dir(self) -> Path:
        return Path(self._config.data_dir)

    @property
    def _build_id(self) -> str:
        return self._config.build_id

    @property
    def _engine_id(self) -> str:
        return self._config.engine_id

    @property
    def _shared_file_uri(self) -> str:
        shared_file_uri: str = self._config.shared_file_uri
        return shared_file_uri.rstrip("/")

    @abstractmethod
    def _download_file(self, path: str, cache: bool = False) -> Path:
        ...

    @abstractmethod
    def _download_folder(self, path: str, cache: bool = False) -> Path:
        ...

    @abstractmethod
    def _exists_file(self, path: str) -> bool:
        ...

    @abstractmethod
    def _upload_file(self, path: str, local_file_path: Path) -> None:
        ...

    @abstractmethod
    def _upload_folder(self, path: str, local_folder_path: Path) -> None:
        ...

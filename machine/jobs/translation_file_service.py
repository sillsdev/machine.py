from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, TypedDict, Union

import json_stream

from machine.jobs.shared_file_service_base import DictToJsonWriter, SharedFileServiceBase
from machine.jobs.shared_file_service_factory import SharedFileServiceType, get_shared_file_service

from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..utils.context_managed_generator import ContextManagedGenerator


class PretranslationInfo(TypedDict):
    corpusId: str  # noqa: N815
    textId: str  # noqa: N815
    refs: List[str]
    translation: str


class TranslationFileService:
    def __init__(
        self,
        type: Union[str, SharedFileServiceType],
        config: Any,
        source_filename: str = "train.src.txt",
        target_filename: str = "train.trg.txt",
        source_pretranslate_filename: str = "pretranslate.src.json",
        target_pretranslate_filename: str = "pretranslate.trg.json",
    ) -> None:

        self._source_filename = source_filename
        self._target_filename = target_filename
        self._source_pretranslate_filename = source_pretranslate_filename
        self._target_pretranslate_filename = target_pretranslate_filename

        self.shared_file_service: SharedFileServiceBase = get_shared_file_service(type, config)

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{self._source_filename}")
        )

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{self._target_filename}")
        )

    def exists_source_corpus(self) -> bool:
        return self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{self._source_filename}")

    def exists_target_corpus(self) -> bool:
        return self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{self._target_filename}")

    def get_source_pretranslations(self) -> ContextManagedGenerator[PretranslationInfo, None, None]:
        src_pretranslate_path = self.shared_file_service.download_file(
            f"{self.shared_file_service.build_path}/{self._source_pretranslate_filename}"
        )

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

    def save_model(self, model_path: Path, destination: str) -> None:
        self.shared_file_service.upload_path(model_path, destination)

    @contextmanager
    def open_target_pretranslation_writer(self) -> Iterator[DictToJsonWriter]:
        return self.shared_file_service.open_target_writer(self._target_pretranslate_filename)

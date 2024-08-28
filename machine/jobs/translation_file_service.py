from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, TypedDict

import json_stream

from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..utils.context_managed_generator import ContextManagedGenerator
from .shared_file_service_base import DictToJsonWriter, SharedFileServiceBase
from .shared_file_service_factory import SharedFileServiceType, get_shared_file_service


class PretranslationInfo(TypedDict):
    corpusId: str  # noqa: N815
    textId: str  # noqa: N815
    refs: List[str]
    translation: str


SOURCE_FILENAME = "train.src.txt"
TARGET_FILENAME = "train.trg.txt"
SOURCE_PRETRANSLATION_FILENAME = "pretranslate.src.json"
TARGET_PRETRANSLATION_FILENAME = "pretranslate.trg.json"


class TranslationFileService:
    def __init__(
        self,
        type: SharedFileServiceType,
        config: Any,
    ) -> None:

        self.shared_file_service: SharedFileServiceBase = get_shared_file_service(type, config)

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{SOURCE_FILENAME}")
        )

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{TARGET_FILENAME}")
        )

    def exists_source_corpus(self) -> bool:
        return self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{SOURCE_FILENAME}")

    def exists_target_corpus(self) -> bool:
        return self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{TARGET_FILENAME}")

    def get_source_pretranslations(self) -> ContextManagedGenerator[PretranslationInfo, None, None]:
        src_pretranslate_path = self.shared_file_service.download_file(
            f"{self.shared_file_service.build_path}/{SOURCE_PRETRANSLATION_FILENAME}"
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
        return self.shared_file_service.open_target_writer(TARGET_PRETRANSLATION_FILENAME)

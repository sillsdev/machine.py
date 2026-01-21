from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, TypedDict

import json_stream

from ..corpora.data_type import DataType
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
    sourceTokens: List[str]  # noqa: N815
    translationTokens: List[str]  # noqa: N815
    alignment: str


class TranslationFileService:
    def __init__(
        self,
        type: SharedFileServiceType,
        config: Any,
        source_filename: Optional[str] = "train.src.txt",
        target_filename: Optional[str] = "train.trg.txt",
        source_terms_filename: Optional[str] = "train.key-terms.src.txt",
        target_terms_filename: Optional[str] = "train.key-terms.trg.txt",
        source_pretranslation_filename: str = "pretranslate.src.json",
        target_pretranslation_filename: str = "pretranslate.trg.json",
    ) -> None:

        self._source_filename = source_filename
        self._target_filename = target_filename
        self._source_terms_filename = source_terms_filename
        self._target_terms_filename = target_terms_filename
        self._source_pretranslation_filename = source_pretranslation_filename
        self._target_pretranslation_filename = target_pretranslation_filename

        self.shared_file_service: SharedFileServiceBase = get_shared_file_service(type, config)

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            file_patterns=[
                self.shared_file_service.download_file(
                    f"{self.shared_file_service.build_path}/{self._source_filename}"
                ),
                self.shared_file_service.download_file(
                    f"{self.shared_file_service.build_path}/{self._source_terms_filename}"
                ),
            ],
            data_types=[DataType.SENTENCE, DataType.GLOSS],
        )

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            file_patterns=[
                self.shared_file_service.download_file(
                    f"{self.shared_file_service.build_path}/{self._target_filename}"
                ),
                self.shared_file_service.download_file(
                    f"{self.shared_file_service.build_path}/{self._target_terms_filename}"
                ),
            ],
            data_types=[DataType.SENTENCE, DataType.GLOSS],
        )

    def exists_source_corpus(self) -> bool:
        return all(
            self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{source_filename}")
            for source_filename in [self._source_filename, self._source_terms_filename]
        )

    def exists_target_corpus(self) -> bool:
        return all(
            self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{target_filename}")
            for target_filename in [self._target_filename, self._target_terms_filename]
        )

    def get_source_pretranslations(self) -> ContextManagedGenerator[PretranslationInfo, None, None]:
        src_pretranslate_path = self.shared_file_service.download_file(
            f"{self.shared_file_service.build_path}/{self._source_pretranslation_filename}"
        )

        def generator() -> Generator[PretranslationInfo, None, None]:
            with src_pretranslate_path.open("r", encoding="utf-8-sig") as file:
                for pi in json_stream.load(file):
                    yield PretranslationInfo(
                        corpusId=pi["corpusId"],
                        textId=pi["textId"],
                        refs=list(pi["refs"]),
                        translation=pi["translation"],
                        sourceTokens=list(),
                        translationTokens=list(),
                        alignment="",
                    )

        return ContextManagedGenerator(generator())

    def save_model(self, model_path: Path, destination: str) -> None:
        self.shared_file_service.upload_path(model_path, destination)

    @contextmanager
    def open_target_pretranslation_writer(self) -> Iterator[DictToJsonWriter]:
        return self.shared_file_service.open_target_writer(self._target_pretranslation_filename)

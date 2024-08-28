from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from .shared_file_service_base import DictToJsonWriter, SharedFileServiceBase
from .shared_file_service_factory import SharedFileServiceType, get_shared_file_service


class WordAlignmentFileService:
    def __init__(
        self,
        type: SharedFileServiceType,
        config: Any,
        source_filename: str = "train.src.txt",
        target_filename: str = "train.trg.txt",
        word_alignment_filename: str = "word_alignments.json",
    ) -> None:

        self._source_filename = source_filename
        self._target_filename = target_filename
        self._word_alignment_filename = word_alignment_filename

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

    def save_model(self, model_path: Path, destination: str) -> None:
        self.shared_file_service.upload_path(model_path, destination)

    @contextmanager
    def open_target_alignment_writer(self) -> Iterator[DictToJsonWriter]:
        return self.shared_file_service.open_target_writer(self._word_alignment_filename)

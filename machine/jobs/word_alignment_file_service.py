from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, TypedDict

import json_stream

from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from .shared_file_service_base import DictToJsonWriter, SharedFileServiceBase
from .shared_file_service_factory import SharedFileServiceType, get_shared_file_service


class WordAlignmentInput(TypedDict):
    corpusId: str  # noqa: N815
    textId: str  # noqa: N815
    refs: List[str]
    source: str
    target: str


class WordAlignmentFileService:
    def __init__(
        self,
        type: SharedFileServiceType,
        config: Any,
        source_filename: str = "train.src.txt",
        target_filename: str = "train.trg.txt",
        word_alignment_input_filename: str = "word_alignments.inputs.json",
        word_alignment_output_filename: str = "word_alignments.outputs.json",
    ) -> None:

        self._source_filename = source_filename
        self._target_filename = target_filename
        self._word_alignment_input_filename = word_alignment_input_filename
        self._word_alignment_output_filename = word_alignment_output_filename

        self.shared_file_service: SharedFileServiceBase = get_shared_file_service(type, config)

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{self._source_filename}")
        )

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{self._target_filename}")
        )

    def get_word_alignment_inputs(self) -> List[WordAlignmentInput]:
        src_pretranslate_path = self.shared_file_service.download_file(
            f"{self.shared_file_service.build_path}/{self._word_alignment_input_filename}"
        )
        with src_pretranslate_path.open("r", encoding="utf-8-sig") as file:
            wa_inputs = [
                WordAlignmentInput(
                    corpusId=pi["corpusId"],
                    textId=pi["textId"],
                    refs=list(pi["refs"]),
                    source=pi["source"],
                    target=pi["target"],
                )
                for pi in json_stream.load(file)
            ]
        return wa_inputs

    def exists_source_corpus(self) -> bool:
        return self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{self._source_filename}")

    def exists_target_corpus(self) -> bool:
        return self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{self._target_filename}")

    def exists_word_alignment_inputs(self) -> bool:
        return self.shared_file_service._exists_file(
            f"{self.shared_file_service.build_path}/{self._word_alignment_input_filename}"
        )

    def save_model(self, model_path: Path, destination: str) -> None:
        self.shared_file_service.upload_path(model_path, destination)

    @contextmanager
    def open_alignment_output_writer(self) -> Iterator[DictToJsonWriter]:
        return self.shared_file_service.open_target_writer(self._word_alignment_output_filename)

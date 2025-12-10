from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, TypedDict, Union

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
        source_filenames: Optional[Union[str, List[str]]] = None,
        target_filenames: Optional[Union[str, List[str]]] = None,
        word_alignment_input_filename: str = "word_alignments.inputs.json",
        word_alignment_output_filename: str = "word_alignments.outputs.json",
    ) -> None:

        if source_filenames is None:
            source_filenames = ["train.src.txt", "train.key-terms.src.txt"]
        if target_filenames is None:
            target_filenames = ["train.trg.txt", "train.key-terms.trg.txt"]

        self._source_filenames = [source_filenames] if isinstance(source_filenames, str) else list(source_filenames)
        self._target_filenames = [target_filenames] if isinstance(target_filenames, str) else list(target_filenames)
        self._word_alignment_input_filename = word_alignment_input_filename
        self._word_alignment_output_filename = word_alignment_output_filename

        self.shared_file_service: SharedFileServiceBase = get_shared_file_service(type, config)

    def create_source_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{source_filename}")
            for source_filename in self._source_filenames
        )

    def create_target_corpus(self) -> TextCorpus:
        return TextFileTextCorpus(
            self.shared_file_service.download_file(f"{self.shared_file_service.build_path}/{target_filename}")
            for target_filename in self._target_filenames
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
        return all(
            self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{source_filename}")
            for source_filename in self._source_filenames
        )

    def exists_target_corpus(self) -> bool:
        return all(
            self.shared_file_service._exists_file(f"{self.shared_file_service.build_path}/{target_filename}")
            for target_filename in self._target_filenames
        )

    def exists_word_alignment_inputs(self) -> bool:
        return self.shared_file_service._exists_file(
            f"{self.shared_file_service.build_path}/{self._word_alignment_input_filename}"
        )

    def save_model(self, model_path: Path, destination: str) -> None:
        self.shared_file_service.upload_path(model_path, destination)

    @contextmanager
    def open_alignment_output_writer(self) -> Iterator[DictToJsonWriter]:
        return self.shared_file_service.open_target_writer(self._word_alignment_output_filename)

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, List, TextIO

import json_stream
from clearml import Task

from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.translation_engine import TranslationEngine
from ..utils import merge_dict
from ..utils.canceled_error import CanceledError
from .nmt_model_factory import NmtModelFactory
from .open_nmt_model_factory import OpenNmtModelFactory
from .shared_file_service import SharedFileService

_PRETRANSLATE_BATCH_SIZE = 128


class ClearMLNmtEngineBuildJob:
    def __init__(
        self, config: dict, nmt_model_factory: NmtModelFactory, shared_file_service: SharedFileService
    ) -> None:
        self._config = config
        self._nmt_model_factory = nmt_model_factory
        self._shared_file_service = shared_file_service

    def run(self, check_canceled: Callable[[], None]) -> None:
        check_canceled()

        print("NMT Engine Build Job started")
        print("Config:", self._config)

        try:
            self._nmt_model_factory.init()

            print("Downloading data files")
            src_train_path = self._shared_file_service.download_file("train.src.txt")
            trg_train_path = self._shared_file_service.download_file("train.trg.txt")
            src_pretranslate_path = self._shared_file_service.download_file("pretranslate.src.json")
            source_corpus = TextFileTextCorpus(src_train_path)
            target_corpus = TextFileTextCorpus(trg_train_path)
            parallel_corpus = source_corpus.align_rows(target_corpus)

            check_canceled()

            print("Training source tokenizer")
            source_tokenizer_trainer = self._nmt_model_factory.create_source_tokenizer_trainer(source_corpus)
            source_tokenizer_trainer.train()
            source_tokenizer_trainer.save()

            check_canceled()

            print("Training target tokenizer")
            target_tokenizer_trainer = self._nmt_model_factory.create_target_tokenizer_trainer(target_corpus)
            target_tokenizer_trainer.train()
            target_tokenizer_trainer.save()

            check_canceled()

            print("Training NMT model")
            model_trainer = self._nmt_model_factory.create_model_trainer(
                self._config["src_lang"], self._config["trg_lang"], parallel_corpus
            )

            model_trainer.train(check_canceled=check_canceled)
            model_trainer.save()

            check_canceled()

            print("Pretranslating segments")
            source_tokenizer = self._nmt_model_factory.create_source_tokenizer()
            target_detokenizer = self._nmt_model_factory.create_target_detokenizer()
            trg_pretranslate_path = self._shared_file_service.data_dir / "pretranslate.trg.json"
            with ExitStack() as stack:
                model = stack.enter_context(self._nmt_model_factory.create_model())
                engine = stack.enter_context(model.create_engine())
                in_file = stack.enter_context(src_pretranslate_path.open("r", encoding="utf-8-sig"))
                out_file = stack.enter_context(trg_pretranslate_path.open("w", encoding="utf-8", newline="\n"))
                src_pretranslate = json_stream.load(in_file)
                out_file.write("[\n")
                batch: List[dict] = []
                first_batch = True
                for pi in src_pretranslate:
                    batch.append(
                        {
                            "corpusId": pi["corpusId"],
                            "textId": pi["textId"],
                            "refs": list(pi["refs"]),
                            "segment": pi["segment"],
                        }
                    )
                    if len(batch) == _PRETRANSLATE_BATCH_SIZE:
                        check_canceled()
                        if not first_batch:
                            out_file.write(",\n")
                        _translate_batch(engine, batch, source_tokenizer, target_detokenizer, out_file)
                        first_batch = False
                        batch.clear()
                if len(batch) > 0:
                    check_canceled()
                    if not first_batch:
                        out_file.write(",\n")
                    _translate_batch(engine, batch, source_tokenizer, target_detokenizer, out_file)
                    first_batch = False
                    batch.clear()
                out_file.write("\n]\n")

                check_canceled()

                print("Uploading pretranslations")
                self._shared_file_service.upload_file("pretranslate.trg.json")
        finally:
            print("Cleaning up")
            self._nmt_model_factory.cleanup()


def _translate_batch(
    engine: TranslationEngine,
    batch: List[dict],
    source_tokenizer: Tokenizer[str, int, str],
    target_detokenizer: Detokenizer[str, str],
    out_file: TextIO,
) -> None:
    source_segments = (list(source_tokenizer.tokenize(pi["segment"])) for pi in batch)
    for i, result in enumerate(engine.translate_batch(source_segments)):
        batch[i]["segment"] = target_detokenizer.detokenize(result.target_segment)
        out_file.write("    " + json.dumps(batch[i]))
        if i < len(batch) - 1:
            out_file.write(",\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--src-lang", required=True, type=str, help="Source language tag")
    parser.add_argument("--trg-lang", required=True, type=str, help="Target language tag")
    parser.add_argument("--build-uri-scheme", required=True, type=str, help="Build URI scheme")
    parser.add_argument("--build-uri", required=True, type=str, help="Build URI")
    args = parser.parse_args()

    task = Task.init()

    def check_canceled() -> None:
        if task.get_status() in {"stopped", "stopping"}:
            raise CanceledError

    config_path = Path(__file__).parent / "config.json"
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)

    merge_dict()
    config = vars(args)
    config["build_id"] = task.name

    shared_file_service = SharedFileService(config)
    nmt_model_factory = OpenNmtModelFactory(config)
    job = ClearMLNmtEngineBuildJob(config, nmt_model_factory, shared_file_service)
    job.run(check_canceled)


if __name__ == "__main__":
    main()

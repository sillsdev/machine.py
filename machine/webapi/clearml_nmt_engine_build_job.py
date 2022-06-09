import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, List, Optional

from clearml import Task

from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.translation_engine import TranslationEngine
from ..utils import merge_dict
from ..utils.canceled_error import CanceledError
from .nmt_model_factory import NmtModelFactory
from .open_nmt_model_factory import OpenNmtModelFactory
from .shared_file_service import PretranslationInfo, PretranslationWriter, SharedFileService

_PRETRANSLATE_BATCH_SIZE = 128


class ClearMLNmtEngineBuildJob:
    def __init__(
        self, config: dict, nmt_model_factory: NmtModelFactory, shared_file_service: SharedFileService
    ) -> None:
        self._config = config
        self._nmt_model_factory = nmt_model_factory
        self._shared_file_service = shared_file_service

    def run(self, check_canceled: Optional[Callable[[], None]] = None) -> None:
        if check_canceled is not None:
            check_canceled()

        print("NMT Engine Build Job started")
        print("Config:", self._config)

        self._nmt_model_factory.init()
        self._shared_file_service.init()

        print("Downloading data files")
        source_corpus = self._shared_file_service.create_source_corpus()
        target_corpus = self._shared_file_service.create_target_corpus()
        parallel_corpus = source_corpus.align_rows(target_corpus)

        if check_canceled is not None:
            check_canceled()

        print("Training source tokenizer")
        source_tokenizer_trainer = self._nmt_model_factory.create_source_tokenizer_trainer(source_corpus)
        source_tokenizer_trainer.train()
        source_tokenizer_trainer.save()

        if check_canceled is not None:
            check_canceled()

        print("Training target tokenizer")
        target_tokenizer_trainer = self._nmt_model_factory.create_target_tokenizer_trainer(target_corpus)
        target_tokenizer_trainer.train()
        target_tokenizer_trainer.save()

        if check_canceled is not None:
            check_canceled()

        print("Training NMT model")
        model_trainer = self._nmt_model_factory.create_model_trainer(
            self._config["src_lang"], self._config["trg_lang"], parallel_corpus
        )

        model_trainer.train(check_canceled=check_canceled)
        model_trainer.save()

        if check_canceled is not None:
            check_canceled()

        print("Pretranslating segments")
        source_tokenizer = self._nmt_model_factory.create_source_tokenizer()
        target_detokenizer = self._nmt_model_factory.create_target_detokenizer()
        with ExitStack() as stack:
            model = stack.enter_context(self._nmt_model_factory.create_model())
            engine = stack.enter_context(model.create_engine())
            src_pretranslations = stack.enter_context(self._shared_file_service.get_source_pretranslations())
            writer = stack.enter_context(self._shared_file_service.open_target_pretranslation_writer())
            batch: List[PretranslationInfo] = []
            for pi in src_pretranslations:
                batch.append(pi)
                if len(batch) == _PRETRANSLATE_BATCH_SIZE:
                    if check_canceled is not None:
                        check_canceled()
                    _translate_batch(engine, batch, source_tokenizer, target_detokenizer, writer)
                    batch.clear()
            if len(batch) > 0:
                if check_canceled is not None:
                    check_canceled()
                _translate_batch(engine, batch, source_tokenizer, target_detokenizer, writer)


def _translate_batch(
    engine: TranslationEngine,
    batch: List[PretranslationInfo],
    source_tokenizer: Tokenizer[str, int, str],
    target_detokenizer: Detokenizer[str, str],
    writer: PretranslationWriter,
) -> None:
    source_segments = (list(source_tokenizer.tokenize(pi["segment"])) for pi in batch)
    for i, result in enumerate(engine.translate_batch(source_segments)):
        batch[i]["segment"] = target_detokenizer.detokenize(result.target_segment)
        writer.write(batch[i])


def run(args: dict) -> None:
    task = Task.init()

    def check_canceled() -> None:
        if task.get_status() in {"stopped", "stopping"}:
            raise CanceledError

    config_path = Path(__file__).parent / "config.json"
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)

    merge_dict(config, args)
    config["build_id"] = task.name

    shared_file_service = SharedFileService(config)
    nmt_model_factory = OpenNmtModelFactory(config)
    job = ClearMLNmtEngineBuildJob(config, nmt_model_factory, shared_file_service)
    job.run(check_canceled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--src-lang", required=True, type=str, help="Source language tag")
    parser.add_argument("--trg-lang", required=True, type=str, help="Target language tag")
    parser.add_argument("--build-uri", required=True, type=str, help="Build URI")
    parser.add_argument("--max-step", type=int, help="Maximum number of steps")
    args = parser.parse_args()

    run(vars(args))


if __name__ == "__main__":
    main()

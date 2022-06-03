import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, TextIO

import json_stream
from clearml import StorageManager, Task

from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.translation_engine import TranslationEngine
from .nmt_model_factory import NmtModelFactory
from .open_nmt_model_factory import OpenNmtModelFactory

_PRETRANSLATE_BATCH_SIZE = 128


class ClearMLNmtEngineBuildJob:
    def __init__(self, config: dict, nmt_model_factory: NmtModelFactory) -> None:
        self._config = config
        self._nmt_model_factory = nmt_model_factory

    def run(self) -> None:
        task = Task.init()

        print("NMT Engine Build Job started")
        print("Config:", self._config)

        try:
            self._nmt_model_factory.init(task.name)

            print("Downloading data files")
            build_uri: str = self._config["build_uri_scheme"] + "://" + self._config["build_uri"]
            build_uri = build_uri.rstrip("/")
            src_train_path = Path(StorageManager.get_local_copy(f"{build_uri}/src.train.txt"))
            trg_train_path = Path(StorageManager.get_local_copy(f"{build_uri}/trg.train.txt"))
            src_pretranslate_path = Path(StorageManager.get_local_copy(f"{build_uri}/src.pretranslate.json"))
            source_corpus = TextFileTextCorpus(src_train_path)
            target_corpus = TextFileTextCorpus(trg_train_path)
            parallel_corpus = source_corpus.align_rows(target_corpus)

            print("Training source tokenizer")
            source_tokenizer_trainer = self._nmt_model_factory.create_source_tokenizer_trainer(task.name, source_corpus)
            source_tokenizer_trainer.train()
            source_tokenizer_trainer.save()

            print("Training target tokenizer")
            target_tokenizer_trainer = self._nmt_model_factory.create_target_tokenizer_trainer(task.name, target_corpus)
            target_tokenizer_trainer.train()
            target_tokenizer_trainer.save()

            print("Training NMT model")
            model_trainer = self._nmt_model_factory.create_model_trainer(
                task.name, self._config["src_lang"], self._config["trg_lang"], parallel_corpus
            )
            model_trainer.train()
            model_trainer.save()

            print("Pretranslating segments")
            source_tokenizer = self._nmt_model_factory.create_source_tokenizer(task.name)
            target_detokenizer = self._nmt_model_factory.create_target_detokenizer(task.name)
            with ExitStack() as stack:
                temp_dir = Path(stack.enter_context(TemporaryDirectory()))
                trg_pretranslate_path = temp_dir / "trg.pretranslate.json"
                model = stack.enter_context(self._nmt_model_factory.create_model(task.name))
                engine = stack.enter_context(model.create_engine())
                in_file = stack.enter_context(src_pretranslate_path.open("r", encoding="utf-8-sig"))
                out_file = stack.enter_context(trg_pretranslate_path.open("w", encoding="utf-8", newline="\n"))
                src_pretranslate = json_stream.load(in_file)
                out_file.write("[\n")
                batch: List[dict] = []
                pi: dict
                for pi in src_pretranslate:
                    batch.append(pi)
                    if len(batch) == _PRETRANSLATE_BATCH_SIZE:
                        _translate_batch(engine, batch, source_tokenizer, target_detokenizer, out_file)
                        batch.clear()
                if len(batch) > 0:
                    _translate_batch(engine, batch, source_tokenizer, target_detokenizer, out_file)
                    batch.clear()
                out_file.write("]\n")

                print("Uploading pretranslations")
                StorageManager.upload_file(str(src_pretranslate_path), f"{build_uri}/src.pretranslate.json")
        finally:
            print("Cleaning up")
            self._nmt_model_factory.cleanup(task.name)


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
        out_file.write("    " + json.dumps(batch[i]) + ",\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--src-lang", required=True, type=str, help="Source language tag")
    parser.add_argument("--trg-lang", required=True, type=str, help="Target language tag")
    parser.add_argument("--build-uri-scheme", required=True, type=str, help="Build URI scheme")
    parser.add_argument("--build-uri", required=True, type=str, help="Build URI")
    args = parser.parse_args()

    config = vars(args)
    config["models_dir"] = "/var/lib/machine/models"
    config["parent_models_dir"] = "/var/lib/machine/parent_models"
    config["data_dir"] = "/var/lib/machine/data"
    config["model"] = "TransformerBase"
    config["mixed_precision"] = True

    nmt_model_factory = OpenNmtModelFactory(config)
    job = ClearMLNmtEngineBuildJob(config, nmt_model_factory)
    job.run()


if __name__ == "__main__":
    main()

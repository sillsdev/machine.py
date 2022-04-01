import shutil
from pathlib import Path
from typing import Tuple

from opennmt import Runner
from opennmt.config import load_model_from_catalog
from opennmt.models import Model


def model_exists(model_dir: str) -> bool:
    model_path = Path(model_dir)
    checkpoint_path = model_path / "checkpoint"
    return checkpoint_path.is_file()


def move_model(src_dir: str, trg_dir: str) -> None:
    src_path = Path(src_dir)
    trg_path = Path(trg_dir)
    checkpoint_path = src_path / "checkpoint"
    checkpoint_path.rename(trg_path / "checkpoint")
    for p in src_path.glob("ckpt-*"):
        p.rename(trg_path / p.name)

    export_dir = src_path / "export"
    if export_dir.is_dir():
        export_dir.rename(trg_path / "export")
    eval_dir = src_path / "eval"
    if eval_dir.is_dir():
        eval_dir.rename(trg_path / "eval")
    avg_dir = src_path / "avg"
    if avg_dir.is_dir():
        avg_dir.rename(trg_path / "avg")
    parent_checkpoint_path = src_path / "parent"
    if parent_checkpoint_path.is_dir():
        parent_checkpoint_path.rename(trg_path / "parent")


def delete_model(model_dir: str) -> None:
    model_path = Path(model_dir)
    checkpoint_path = model_path / "checkpoint"
    if checkpoint_path.is_file():
        checkpoint_path.unlink()
    for p in model_path.glob("ckpt-*"):
        p.unlink()

    export_dir = model_path / "export"
    if export_dir.is_dir():
        shutil.rmtree(export_dir)
    eval_dir = model_path / "eval"
    if eval_dir.is_dir():
        shutil.rmtree(eval_dir)
    avg_dir = model_path / "avg"
    if avg_dir.is_dir():
        shutil.rmtree(avg_dir)
    parent_checkpoint_path = model_path / "parent"
    if parent_checkpoint_path.is_dir():
        shutil.rmtree(parent_checkpoint_path)


def delete_train_summary_files(model_dir: str) -> None:
    model_path = Path(model_dir)
    for p in model_path.glob("**/events.out.*"):
        p.unlink()
    for p in model_path.glob("model_examples_inputter_*.txt"):
        p.unlink()
    projector_config_path = model_path / "projector_config.pbtxt"
    if projector_config_path.is_file():
        projector_config_path.unlink()


def delete_corpus_files(config: dict) -> None:
    data_config: dict = config["data"]
    train_src_path = Path(data_config["train_features_file"])
    train_trg_path = Path(data_config["train_labels_file"])
    eval_src_path = Path(data_config["eval_features_file"])
    eval_trg_path = Path(data_config["eval_labels_file"])
    if train_src_path.is_file():
        train_src_path.unlink()
    if train_trg_path.is_file():
        train_trg_path.unlink()
    if eval_src_path.is_file():
        eval_src_path.unlink()
    if eval_trg_path.is_file():
        eval_trg_path.unlink()


def move_corpus_files(src_config: dict, trg_config: dict) -> None:
    _move_corpus_file(src_config, trg_config, "train_features_file")
    _move_corpus_file(src_config, trg_config, "train_labels_file")
    _move_corpus_file(src_config, trg_config, "eval_features_file")
    _move_corpus_file(src_config, trg_config, "eval_labels_file")


def _move_corpus_file(src_config: dict, trg_config: dict, name: str) -> None:
    model_dir = Path(src_config["model_dir"])
    src_data_config = src_config["data"]
    trg_data_config = trg_config["data"]
    path = Path(src_data_config[name])
    if model_dir in path.parents:
        path.rename(trg_data_config[name])


class OpenNmtRunner(Runner):
    def __init__(
        self, model_type: str, config: dict, mixed_precision: bool = False, prefix_corpus_paths: bool = False
    ) -> None:
        super().__init__(load_model_from_catalog(model_type), config, mixed_precision=mixed_precision)
        self._model_type = model_type
        if prefix_corpus_paths:
            data_config = self._config["data"]
            self._prefix_path(data_config, "train_features_file")
            self._prefix_path(data_config, "train_labels_file")
            self._prefix_path(data_config, "eval_features_file")
            self._prefix_path(data_config, "eval_labels_file")

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def config(self) -> dict:
        return self._config

    @property
    def mixed_precision(self) -> bool:
        return self._mixed_precision

    def load(self, training: bool = False) -> Tuple[dict, Model]:
        config = self._finalize_config(training)
        model = self._init_model(config)
        return config, model

    def _prefix_path(self, data_config: dict, name: str) -> None:
        if name not in data_config:
            return
        path = Path(data_config[name])
        if not path.is_absolute():
            path = Path(self.model_dir) / path
            data_config[name] = str(path)

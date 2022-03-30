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


def delete_train_summary_files(model_dir: str) -> None:
    model_path = Path(model_dir)
    for p in model_path.glob("events.out.*"):
        p.unlink()
    for p in model_path.glob("model_examples_inputter_*.txt"):
        p.unlink()
    projector_config_path = model_path / "projector_config.pbtxt"
    if projector_config_path.is_file():
        projector_config_path.unlink()


class OpenNmtRunner(Runner):
    def __init__(self, model_type: str, config: dict, mixed_precision: bool = False):
        super().__init__(load_model_from_catalog(model_type), config, mixed_precision=mixed_precision)
        self._model_type = model_type

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

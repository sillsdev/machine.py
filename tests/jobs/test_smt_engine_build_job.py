from pathlib import Path
from tempfile import mkdtemp

from pytest import raises

from machine.jobs.build_smt_engine import SmtEngineBuildJob
from machine.jobs.config import SETTINGS
from machine.jobs.local_shared_file_service import LocalSharedFileService
from machine.utils import CanceledError
from machine.utils.progress_status import ProgressStatus


def test_run() -> None:
    env = _TestEnvironment()
    env.run()
    assert env.check_files_created()


def test_cancel() -> None:
    env = _TestEnvironment()
    env.cancel_job = True
    raises(CanceledError, env.run)


class _TestEnvironment:
    def __init__(self) -> None:
        self.cancel_job = False
        self.percent_completed = 0
        self.build_id = "build1"
        self.model_name = "myModelName"
        temp_dir = mkdtemp()
        self.temp_path = Path(temp_dir)
        self.setup_corpus()
        config = {
            "model_type": "hmm",
            "build_id": self.build_id,
            "save_model": self.model_name,
            "shared_file_uri": temp_dir,
        }
        SETTINGS.update(config)

        shared_file_service = LocalSharedFileService(SETTINGS)

        self.job = SmtEngineBuildJob(SETTINGS, shared_file_service)

    def run(self):
        self.job.run(progress=self.progress, check_canceled=self.check_canceled)

    def setup_corpus(self):
        train_target_path = self.temp_path / "builds" / self.build_id / "train.trg.txt"
        train_target_path.parent.mkdir(parents=True, exist_ok=True)
        with train_target_path.open("w+", encoding="utf-8") as f:
            f.write(
                """Would you mind giving us the keys to the room, please?
I have made a reservation for a quiet, double room with a telephone and a tv for Rosario Cabedo."""
            )
        train_source_path = self.temp_path / "builds" / self.build_id / "train.src.txt"
        with train_source_path.open("w+", encoding="utf-8") as f:
            f.write(
                """¿Le importaría darnos las llaves de la habitación, por favor?
He hecho la reserva de una habitación tranquila doble con teléfono y televisión a nombre de Rosario Cabedo."""
            )

    def check_files_created(self) -> bool:
        model_path = self.temp_path / "models" / f"{self.model_name}.tar.gz"
        return model_path.exists()

    def check_canceled(self) -> None:
        if self.cancel_job:
            raise CanceledError

    def progress(self, status: ProgressStatus) -> None:
        if status.percent_completed is not None:
            self.percent_completed = status.percent_completed

import argparse
import os
from typing import cast

from clearml import Task

from ..utils.canceled_error import CanceledError
from .clearml_shared_file_service import ClearMLSharedFileService
from .config import SETTINGS
from .nmt_engine_build_job import NmtEngineBuildJob
from .nmt_model_factory import NmtModelFactory


def run(args: dict) -> None:
    task = Task.init()

    def check_canceled() -> None:
        if task.get_status() in {"stopped", "stopping"}:
            raise CanceledError

    SETTINGS.update(args)
    SETTINGS.data_dir = os.path.expanduser(cast(str, SETTINGS.data_dir))

    shared_file_service = ClearMLSharedFileService(SETTINGS)
    model_type = cast(str, SETTINGS.model_type).lower()
    nmt_model_factory: NmtModelFactory
    if model_type == "huggingface":
        from .huggingface.hugging_face_nmt_model_factory import HuggingFaceNmtModelFactory

        nmt_model_factory = HuggingFaceNmtModelFactory(SETTINGS, shared_file_service)
    elif model_type == "opennmt":
        from .opennmt.open_nmt_model_factory import OpenNmtModelFactory

        nmt_model_factory = OpenNmtModelFactory(SETTINGS, shared_file_service)
    else:
        raise RuntimeError("The model type is invalid.")

    print("NMT Engine Build Job started")
    print("Config: ", SETTINGS.as_dict())
    job = NmtEngineBuildJob(SETTINGS, nmt_model_factory, shared_file_service)
    job.run(check_canceled)
    print("Finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model.")
    parser.add_argument("--model-type", required=True, type=str, help="Model type")
    parser.add_argument("--engine-id", required=True, type=str, help="Engine id")
    parser.add_argument("--build-id", required=True, type=str, help="Build id")
    parser.add_argument("--src-lang", required=True, type=str, help="Source language tag")
    parser.add_argument("--trg-lang", required=True, type=str, help="Target language tag")
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps")
    args = parser.parse_args()

    run(vars(args))


if __name__ == "__main__":
    main()

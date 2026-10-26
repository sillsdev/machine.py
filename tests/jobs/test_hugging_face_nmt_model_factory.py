import sys

if sys.platform == "darwin":
    from pytest import skip

    skip("skipping Hugging Face tests on MacOS", allow_module_level=True)

from typing import Optional

from pytest import mark
from testutils.mock_settings import MockSettings

from machine.jobs.huggingface.hugging_face_nmt_model_factory import HuggingFaceNmtModelFactory


@mark.parametrize(
    "group_by_length, train_sampling_strategy, expected",
    [
        (None, "group_by_length", "group_by_length"),
        (True, "group_by_length", "group_by_length"),
        (False, "group_by_length", "random"),
        (False, "sequential", "sequential"),
        (True, "random", "random"),
    ],
)
def test_group_by_length_backwards_compatibility(
    group_by_length: Optional[bool], train_sampling_strategy: str, expected: str
) -> None:
    config = MockSettings(
        {
            "data_dir": "/tmp",
            "shared_file_folder": "test",
            "build_id": "build1",
            "clearml": False,
            "huggingface": {
                "train_params": {
                    "group_by_length": group_by_length,
                    "train_sampling_strategy": train_sampling_strategy,
                }
            },
        }
    )

    factory = HuggingFaceNmtModelFactory(config)

    assert factory._training_args.train_sampling_strategy == expected

import logging
from base64 import b64encode

from pytest import raises
from testutils.mock_settings import MockSettings

from machine.jobs.build_clearml_helper import update_settings
from machine.jobs.nmt_build_options import NmtBuildOptions
from machine.jobs.smt_build_options import SmtBuildOptions
from machine.jobs.word_alignment_build_options import WordAlignmentBuildOptions

logger = logging.getLogger(__name__)


def test_update_settings_invalid_build_options() -> None:
    settings = MockSettings({"model_type": "thot", "data_dir": "/tmp"})
    # Invalid JSON
    with raises(ValueError):
        update_settings(settings, {"build_options": "invalid_json"}, None, logger, NmtBuildOptions)
    # Invalid value
    with raises(ValueError):
        update_settings(settings, {"build_options": "0"}, None, logger, NmtBuildOptions)
    # Invalid/unsupported property
    with raises(ValueError):
        update_settings(settings, {"build_options": '{"invalid":"option"}'}, None, logger, NmtBuildOptions)
    # Unsupported model
    with raises(ValueError):
        update_settings(
            settings, {"build_options": '{"parent_model_name":"not_allowed_this"}'}, None, logger, NmtBuildOptions
        )
    # Specifying NMT arguments for SMT
    with raises(ValueError):
        update_settings(
            settings,
            {"build_options": '{"parent_model_name":"hf-internal-testing/tiny-random-nllb"}'},
            None,
            logger,
            SmtBuildOptions,
        )


def test_update_settings_valid_base64_build_options() -> None:
    settings = MockSettings({"model_type": "thot", "data_dir": "/tmp"})
    build_options_base64 = b64encode('{"parent_model_name":"hf-internal-testing/tiny-random-nllb"}'.encode("ascii"))
    update_settings(settings, {"build_options": build_options_base64}, None, logger, NmtBuildOptions)
    assert settings.thot.parent_model_name == "hf-internal-testing/tiny-random-nllb"


def test_update_settings_valid_nmt_build_options() -> None:
    settings = MockSettings({"model_type": "thot", "data_dir": "/tmp"})
    update_settings(
        settings,
        {"build_options": '{"parent_model_name":"hf-internal-testing/tiny-random-nllb","align_pretranslations":true}'},
        None,
        logger,
        NmtBuildOptions,
    )
    assert settings.thot.parent_model_name == "hf-internal-testing/tiny-random-nllb"
    assert settings.align_pretranslations


def test_update_settings_valid_smt_build_options() -> None:
    settings = MockSettings({"model_type": "thot", "data_dir": "/tmp"})
    update_settings(
        settings, {"build_options": '{"thot_mt":{"word_alignment_model_type":"hmm"}}'}, None, logger, SmtBuildOptions
    )
    assert settings.thot_mt.word_alignment_model_type == "hmm"


def test_update_settings_valid_word_alignment_build_options() -> None:
    settings = MockSettings({"model_type": "thot", "data_dir": "/tmp"})
    update_settings(
        settings, {"build_options": '{"thot_align":{"model_type":"hmm"}}'}, None, logger, WordAlignmentBuildOptions
    )
    assert settings.thot_align.model_type == "hmm"

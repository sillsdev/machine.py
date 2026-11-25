import logging
from base64 import b64encode

from pytest import raises
from testutils.mock_settings import MockSettings

from machine.jobs.build_clearml_helper import update_settings
from machine.jobs.build_remote_llm_engine import RemoteLlmBuildOptions
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


def test_post_edit_model_options() -> None:
    for model_type, options, model in (
        ("huggingface", '{"post_edit_model":"openai/gpt-4o"}', NmtBuildOptions),
        ("remote_llm", '{"model":"test","post_edit_model":"openai/gpt-4o"}', RemoteLlmBuildOptions),
    ):
        settings = MockSettings({"model_type": model_type, "data_dir": "/tmp"})
        update_settings(settings, {"build_options": options}, None, logger, model)
        assert settings.post_edit_model == "openai/gpt-4o"


def test_post_edit_model_rejects_blank() -> None:
    settings = MockSettings({"model_type": "huggingface", "data_dir": "/tmp"})
    with raises(ValueError):
        update_settings(settings, {"build_options": '{"post_edit_model":""}'}, None, logger, NmtBuildOptions)


def test_remote_post_edit_model_rejects_blank() -> None:
    settings = MockSettings({"model_type": "remote_llm", "data_dir": "/tmp"})
    with raises(ValueError):
        update_settings(
            settings,
            {"build_options": '{"model":"test","post_edit_model":""}'},
            None,
            logger,
            RemoteLlmBuildOptions,
        )


def test_post_edit_model_does_not_leak_between_builds() -> None:
    settings = MockSettings({"model_type": "huggingface", "data_dir": "/tmp"})
    update_settings(settings, {"build_options": '{"post_edit_model":"openai/gpt-4o"}'}, None, logger, NmtBuildOptions)
    update_settings(settings, {"build_options": "{}"}, None, logger, NmtBuildOptions)
    assert settings.post_edit_model is None


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


def test_update_settings_valid_remote_llm_build_options() -> None:
    settings = MockSettings({"model_type": "remote_llm", "data_dir": "/tmp"})

    update_settings(
        settings,
        {"build_options": '{"model":"openai/gpt-4o","batch_size":2,"num_workers":3}'},
        None,
        logger,
        RemoteLlmBuildOptions,
    )

    assert settings.remote_llm.model == "openai/gpt-4o"
    assert settings.remote_llm.batch_size == 2
    assert settings.remote_llm.num_workers == 3


def test_remote_llm_build_options_accept_custom_prompt() -> None:
    settings = MockSettings({"model_type": "remote_llm", "data_dir": "/tmp"})
    update_settings(
        settings,
        {
            "build_options": '{"model":"test","prompt":{"instruction_template":"{source}",'
            '"batch_instruction_template":"{num_segments}: {source}","example_format":{"type":"text",'
            '"template":"{source} => {target}"}}}'
        },
        None,
        logger,
        RemoteLlmBuildOptions,
    )

    assert settings.remote_llm.prompt.instruction_template == "{source}"
    assert settings.remote_llm.prompt.example_format.template == "{source} => {target}"

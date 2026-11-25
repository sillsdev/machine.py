import argparse
import logging
import os
from typing import Callable, Literal, Optional

from clearml import Task
from pydantic import BaseModel, ConfigDict, Field

from ..translation.llm.example_retriever_method import ExampleRetrievalMethod
from ..translation.llm.openrouter_llm_completion_client import OpenRouterLlmCompletionClient
from ..utils.canceled_error import CanceledError
from ..utils.progress_status import ProgressStatus
from .build_clearml_helper import report_clearml_progress, update_settings
from .config import SETTINGS
from .remote_llm_engine_build_job import RemoteLlmEngineBuildJob
from .remote_llm_model_factory import RemoteLlmModelFactory
from .shared_file_service_factory import SharedFileServiceType
from .thot.thot_word_alignment_model_factory import ThotWordAlignmentModelFactory
from .translation_file_service import TranslationFileService


class RemoteLlmExampleFormat(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["text", "json", "xml"] = "text"
    template: str | None = None
    separator: str | None = None
    escape: bool | None = None


class RemoteLlmPromptOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    system_message: str | None = None
    instruction_template: str | None = None
    batch_system_message: str | None = None
    batch_instruction_template: str | None = None
    example_format: Literal["text", "json", "xml"] | RemoteLlmExampleFormat | None = None
    back_translation: bool = False


class RemoteLlmBuildOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str
    max_tokens: int | None = None
    num_examples: int = Field(default=100, ge=0)
    retrieval_method: Literal["tfidf", "coverage"] = "coverage"
    batch_size: int = Field(default=15, ge=1)
    num_workers: int = Field(default=4, ge=1)
    prompt: RemoteLlmPromptOptions | None = None
    align_pretranslations: bool | None = None
    post_edit_model: str | None = Field(default=None, min_length=1)
    tags: list[str] | str | None = None


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(str(__package__) + ".build_remote_llm_engine")


def run(args: dict) -> None:
    progress: Optional[Callable[[ProgressStatus], None]] = None
    check_canceled: Optional[Callable[[], None]] = None
    task: Optional[Task] = None
    if args["clearml"]:
        task = Task.init()

        def clearml_check_canceled() -> None:
            if task.get_status() == "stopped":
                raise CanceledError

        check_canceled = clearml_check_canceled

        def clearml_progress(status: ProgressStatus) -> None:
            if status.percent_completed is not None:
                report_clearml_progress(
                    task=task, percent_completed=round(status.percent_completed * 100), progress_status=status
                )

        progress = clearml_progress

    try:
        logger.info("Remote LLM Engine Build Job started")
        args["model_type"] = "remote_llm"
        update_settings(SETTINGS, args, task, logger, RemoteLlmBuildOptions)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        translation_file_service = TranslationFileService(SharedFileServiceType.CLEARML, SETTINGS)
        retrieval_method = (
            ExampleRetrievalMethod.TFIDF
            if SETTINGS.remote_llm.retrieval_method == "tfidf"
            else ExampleRetrievalMethod.COVERAGE
        )
        with OpenRouterLlmCompletionClient(
            model=SETTINGS.remote_llm.model, api_key=api_key, max_tokens=SETTINGS.remote_llm.max_tokens
        ) as completion_client:
            job = RemoteLlmEngineBuildJob(
                SETTINGS,
                RemoteLlmModelFactory(SETTINGS, completion_client, retrieval_method),
                translation_file_service,
                ThotWordAlignmentModelFactory(SETTINGS),
                completion_client,
            )
            train_corpus_size, _ = job.run(progress, check_canceled)
        if task is not None:
            task.get_logger().report_single_value(name="example_corpus_size", value=train_corpus_size)
            usage = completion_client.usage_summary
            task.get_logger().report_single_value(name="llm_request_count", value=usage.request_count)
            task.get_logger().report_single_value(name="llm_prompt_tokens", value=usage.input_token_count)
            task.get_logger().report_single_value(name="llm_completion_tokens", value=usage.output_token_count)
            task.get_logger().report_single_value(name="llm_reported_cost_usd", value=usage.reported_cost)
            task.get_logger().report_single_value(name="llm_unpriced_request_count", value=usage.unpriced_request_count)
        logger.info("Finished")
    except Exception as e:
        if task:
            if task.get_status() == "stopped":
                return
            task.mark_failed(status_reason=type(e).__name__, status_message=str(e))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Builds a remote LLM translation engine.")
    parser.add_argument("--engine-id", required=True, type=str, help="Engine id")
    parser.add_argument("--build-id", required=True, type=str, help="Build id")
    parser.add_argument("--src-lang", required=True, type=str, help="Source language name")
    parser.add_argument("--trg-lang", required=True, type=str, help="Target language name")
    parser.add_argument("--clearml", default=False, action="store_true", help="Initializes a ClearML task")
    parser.add_argument("--build-options", required=True, type=str, help="JSON build configurations")
    args = parser.parse_args()

    run({k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main()

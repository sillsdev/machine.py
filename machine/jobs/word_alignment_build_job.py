import json
import logging
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Optional

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.corpora_utils import batch
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..corpora.token_processors import lowercase
from ..tokenization.tokenizer_factory import create_tokenizer
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .word_alignment_file_service import WordAlignmentFileService, WordAlignmentInput
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


class WordAlignmentBuildJob:
    def __init__(
        self,
        config: Any,
        word_alignment_model_factory: WordAlignmentModelFactory,
        word_alignment_file_service: WordAlignmentFileService,
    ) -> None:
        self._word_alignment_model_factory = word_alignment_model_factory
        self._word_alignment_model_factory.init()
        self._config = config
        self._tokenizer = create_tokenizer(self._config.thot_align.tokenizer)
        self._word_alignment_file_service = word_alignment_file_service
        self._train_corpus_size = -1

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> int:
        if check_canceled is not None:
            check_canceled()

        progress_reporter = self._get_progress_reporter(progress)

        source_corpus = self._word_alignment_file_service.create_source_corpus()
        target_corpus = self._word_alignment_file_service.create_target_corpus()
        parallel_corpus: ParallelTextCorpus = source_corpus.align_rows(target_corpus)

        if parallel_corpus.count(include_empty=False) == 0:
            raise RuntimeError("No parallel corpus data found")

        train_corpus_size = self._train_model(parallel_corpus, progress_reporter, check_canceled)

        if check_canceled is not None:
            check_canceled()

        logger.info("Generating alignments")

        self._batch_inference(progress_reporter, check_canceled)

        self._save_model()
        return train_corpus_size

    def _get_progress_reporter(self, progress: Optional[Callable[[ProgressStatus], None]]) -> PhasedProgressReporter:
        phases = [
            Phase(message="Training Word Alignment model", percentage=0.9, stage="train"),
            Phase(message="Aligning segments", percentage=0.1, stage="inference"),
        ]
        return PhasedProgressReporter(progress, phases)

    def _train_model(
        self,
        parallel_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> int:

        with (
            progress_reporter.start_next_phase() as phase_progress,
            self._word_alignment_model_factory.create_model_trainer(self._tokenizer, parallel_corpus) as trainer,
        ):
            trainer.train(progress=phase_progress, check_canceled=check_canceled)
            trainer.save()
            train_corpus_size = trainer.stats.train_corpus_size

        if check_canceled is not None:
            check_canceled()
        return train_corpus_size

    def _batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:

        inference_inputs = self._word_alignment_file_service.get_word_alignment_inputs()

        inference_step_count = len(inference_inputs)

        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            writer = stack.enter_context(self._word_alignment_file_service.open_alignment_output_writer())
            current_inference_step = 0
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))

            temp_dir = stack.enter_context(TemporaryDirectory())
            # Spool the parallel data to disk so that the aligner can make multiple
            # passes over them without keeping the entire corpus in memory.
            source_path = Path(temp_dir) / "word_align.src.txt"
            target_path = Path(temp_dir) / "word_align.trg.txt"
            word_alignments_path = Path(temp_dir) / "word_align.json"
            with (
                source_path.open("w", encoding="utf-8", newline="\n") as source_file,
                target_path.open("w", encoding="utf-8", newline="\n") as target_file,
                word_alignments_path.open("w", encoding="utf-8", newline="\n") as word_alignments_file,
            ):
                for wa_input in inference_inputs:
                    source_file.write(wa_input["source"] + "\n")
                    target_file.write(wa_input["target"] + "\n")
                    word_alignments_file.write(json.dumps(wa_input, ensure_ascii=False) + "\n")

            if check_canceled is not None:
                check_canceled()

            parallel_corpus = TextFileTextCorpus(source_path).align_rows(TextFileTextCorpus(target_path))

            batch_size: int = self._config["inference_batch_size"]
            alignment_model = stack.enter_context(self._word_alignment_model_factory.create_alignment_model())
            rows = stack.enter_context(parallel_corpus.tokenize(self._tokenizer).get_rows())
            for wa_batch in batch(zip(_read_word_alignments(word_alignments_path), rows, strict=True), batch_size):
                if check_canceled is not None:
                    check_canceled()
                segments = [(lowercase(row.source_segment), lowercase(row.target_segment)) for _, row in wa_batch]
                alignments = alignment_model.align_batch(segments)
                if check_canceled is not None:
                    check_canceled()
                for (wa_input, row), (source_segment, target_segment), alignment in zip(
                    wa_batch, segments, alignments, strict=True
                ):
                    word_pairs = alignment.to_aligned_word_pairs(include_null=False)
                    alignment_model.compute_aligned_word_pair_scores(source_segment, target_segment, word_pairs)

                    word_alignment_info = {
                        "corpusId": wa_input["corpusId"],
                        "textId": wa_input["textId"],
                        "sourceRefs": [str(ref) for ref in wa_input["sourceRefs"]],
                        "targetRefs": [str(ref) for ref in wa_input["targetRefs"]],
                        "sourceTokens": row.source_segment,
                        "targetTokens": row.target_segment,
                        "alignment": AlignedWordPair.to_string(word_pairs),
                    }
                    writer.write(word_alignment_info)

    def _save_model(self) -> None:
        logger.info("Saving model")
        model_path = self._word_alignment_model_factory.save_model()
        self._word_alignment_file_service.save_model(
            model_path, f"builds/{self._config['build_id']}/model{''.join(model_path.suffixes)}"
        )


def _read_word_alignments(path: Path) -> Generator[WordAlignmentInput, None, None]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)

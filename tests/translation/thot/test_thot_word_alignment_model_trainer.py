from pathlib import Path
from tempfile import TemporaryDirectory

from testutils.thot_test_helpers import create_test_parallel_corpus
from translation.thot.thot_model_trainer_helper import get_emtpy_parallel_corpus, get_parallel_corpus

from machine.corpora import ParallelTextCorpus
from machine.tokenization import StringTokenizer, WhitespaceTokenizer
from machine.translation import SymmetrizedWordAlignmentModelTrainer, WordAlignmentMatrix
from machine.translation.thot import (
    ThotFastAlignWordAlignmentModel,
    ThotSymmetrizedWordAlignmentModel,
    ThotWordAlignmentModelTrainer,
    create_thot_symmetrized_word_alignment_model,
    create_thot_word_alignment_model,
)


def train_model(
    corpus: ParallelTextCorpus,
    direct_model_path: Path,
    inverse_model_path: Path,
    thot_word_alignment_model_type: str,
    tokenizer: StringTokenizer,
):
    direct_trainer = ThotWordAlignmentModelTrainer(
        thot_word_alignment_model_type,
        corpus.lowercase(),
        prefix_filename=direct_model_path,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
    )
    inverse_trainer = ThotWordAlignmentModelTrainer(
        thot_word_alignment_model_type,
        corpus.invert().lowercase(),
        prefix_filename=inverse_model_path,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
    )

    with SymmetrizedWordAlignmentModelTrainer(direct_trainer, inverse_trainer) as trainer:
        trainer.train(lambda status: print(f"{status.message}: {status.percent_completed:.2%}"))
        trainer.save()


def test_train_non_empty_corpus() -> None:
    thot_word_alignment_model_type = "hmm"
    tokenizer = WhitespaceTokenizer()
    corpus = get_parallel_corpus()

    with TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        (tmp_path / "tm").mkdir()
        direct_model_path = tmp_path / "tm" / "src_trg_invswm"
        inverse_model_path = tmp_path / "tm" / "src_trg_swm"
        train_model(corpus, direct_model_path, inverse_model_path, thot_word_alignment_model_type, tokenizer)
        with ThotSymmetrizedWordAlignmentModel(
            create_thot_word_alignment_model(thot_word_alignment_model_type, direct_model_path),
            create_thot_word_alignment_model(thot_word_alignment_model_type, inverse_model_path),
        ) as model:
            matrix = model.align(
                list(tokenizer.tokenize("una habitación individual por semana")),
                list(tokenizer.tokenize("a single room cost per week")),
            )
            assert matrix == WordAlignmentMatrix.from_word_pairs(5, 6, {(0, 2), (1, 2), (2, 3), (2, 4), (2, 5)})


def test_train_empty_corpus() -> None:
    thot_word_alignment_model_type = "hmm"
    tokenizer = WhitespaceTokenizer()
    corpus = get_emtpy_parallel_corpus()
    with TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        direct_model_path = tmp_path / "tm" / "src_trg_invswm"
        inverse_model_path = tmp_path / "tm" / "src_trg_swm"
        train_model(corpus, direct_model_path, inverse_model_path, thot_word_alignment_model_type, tokenizer)
        with ThotSymmetrizedWordAlignmentModel(
            create_thot_word_alignment_model(thot_word_alignment_model_type, direct_model_path),
            create_thot_word_alignment_model(thot_word_alignment_model_type, inverse_model_path),
        ) as model:
            matrix = model.align("una habitación individual por semana", "a single room cost per week")
            assert matrix == WordAlignmentMatrix.from_word_pairs(5, 6, {(0, 0)})


def test_emit_training_alignments_single_direction() -> None:
    corpus = create_test_parallel_corpus()
    row = next(iter(corpus.get_rows()))
    model = ThotFastAlignWordAlignmentModel()
    model.emit_training_alignments = True
    with model.create_trainer(corpus) as trainer:
        trainer.train()
        trainer.save()
    assert model.training_alignment_count == 8
    # For a deterministic model, the retained training alignment matches the inference alignment,
    # and it survives the trainer being closed.
    assert model.get_training_alignment(0) == model.align(row.source_segment, row.target_segment)


def test_emit_training_alignments_symmetrized() -> None:
    corpus = create_test_parallel_corpus()
    row = next(iter(corpus.get_rows()))
    model = create_thot_symmetrized_word_alignment_model("fast_align")
    model.emit_training_alignments = True
    with model.create_trainer(corpus) as trainer:
        trainer.train()
        trainer.save()
    assert model.training_alignment_count == 8
    # The C++ symmetrized transductive alignment matches the C++ symmetrized inference alignment.
    assert model.get_training_alignment(0) == model.align(row.source_segment, row.target_segment)


def test_emit_training_alignments_disabled() -> None:
    corpus = create_test_parallel_corpus()
    model = ThotFastAlignWordAlignmentModel()
    with model.create_trainer(corpus) as trainer:
        trainer.train()
        trainer.save()
    # When emission is not enabled, retrieval returns a degenerate result rather than raising.
    alignment = model.get_training_alignment(0)
    assert isinstance(alignment, WordAlignmentMatrix)

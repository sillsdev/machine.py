from pathlib import Path
from tempfile import TemporaryDirectory

from translation.thot.thot_model_trainer_helper import get_emtpy_parallel_corpus, get_parallel_corpus

from machine.corpora.parallel_text_corpus import ParallelTextCorpus
from machine.tokenization import StringTokenizer, WhitespaceTokenizer
from machine.translation.symmetrized_word_alignment_model_trainer import SymmetrizedWordAlignmentModelTrainer
from machine.translation.thot import ThotWordAlignmentModelTrainer
from machine.translation.thot.thot_symmetrized_word_alignment_model import ThotSymmetrizedWordAlignmentModel
from machine.translation.thot.thot_word_alignment_model_utils import create_thot_word_alignment_model
from machine.translation.word_alignment_matrix import WordAlignmentMatrix


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


if __name__ == "__main__":
    test_train_non_empty_corpus()
    test_train_empty_corpus()

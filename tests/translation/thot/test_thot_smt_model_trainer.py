import os
from tempfile import TemporaryDirectory

from translation.thot.thot_model_trainer_helper import get_emtpy_parallel_corpus, get_parallel_corpus

from machine.translation.thot import ThotSmtModel, ThotSmtModelTrainer, ThotSmtParameters, ThotWordAlignmentModelType


def test_train_non_empty_corpus() -> None:
    with TemporaryDirectory() as temp_dir:
        corpus = get_parallel_corpus()

        parameters = ThotSmtParameters(
            translation_model_filename_prefix=os.path.join(temp_dir, "tm", "src_trg"),
            language_model_filename_prefix=os.path.join(temp_dir, "lm", "trg.lm"),
        )

        with ThotSmtModelTrainer(ThotWordAlignmentModelType.HMM, corpus, parameters) as trainer:
            trainer.train()
            trainer.save()
            parameters = trainer.parameters

        with ThotSmtModel(ThotWordAlignmentModelType.HMM, parameters) as model:
            result = model.translate("una habitación individual por semana")
            assert result.translation == "a single room cost per week"


def test_train_empty_corpus() -> None:
    with TemporaryDirectory() as temp_dir:
        corpus = get_emtpy_parallel_corpus()

        parameters = ThotSmtParameters(
            translation_model_filename_prefix=os.path.join(temp_dir, "tm", "src_trg"),
            language_model_filename_prefix=os.path.join(temp_dir, "lm", "trg.lm"),
        )

        with ThotSmtModelTrainer(ThotWordAlignmentModelType.HMM, corpus, parameters) as trainer:
            trainer.train()
            trainer.save()
            parameters = trainer.parameters

        with ThotSmtModel(ThotWordAlignmentModelType.HMM, parameters) as model:
            result = model.translate("una habitación individual por semana")
            assert result.translation == "una habitación individual por semana"

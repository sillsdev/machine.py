import os
from tempfile import TemporaryDirectory

from machine.corpora import (
    AlignedWordPair,
    AlignmentRow,
    DictionaryAlignmentCorpus,
    DictionaryTextCorpus,
    MemoryAlignmentCollection,
    MemoryText,
    TextRow,
)
from machine.translation.thot import ThotSmtModel, ThotSmtModelTrainer, ThotSmtParameters, ThotWordAlignmentModelType


def test_train_non_empty_corpus() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "¿ le importaría darnos las llaves de la habitación , por favor ?"),
                        _row(
                            2,
                            "he hecho la reserva de una habitación tranquila doble con ||| teléfono ||| y televisión a "
                            "nombre de rosario cabedo .",
                        ),
                        _row(3, "¿ le importaría cambiarme a otra habitación más tranquila ?"),
                        _row(4, "por favor , tengo reservada una habitación ."),
                        _row(5, "me parece que existe un problema ."),
                        _row(6, "¿ tiene habitaciones libres con televisión , aire acondicionado y caja fuerte ?"),
                        _row(7, "¿ le importaría mostrarnos una habitación con televisión ?"),
                        _row(8, "¿ tiene teléfono ?"),
                        _row(9, "voy a marcharme el dos a las ocho de la noche ."),
                        _row(10, "¿ cuánto cuesta una habitación individual por semana ?"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "would you mind giving us the keys to the room , please ?"),
                        _row(
                            2,
                            "i have made a reservation for a quiet , double room with a ||| telephone ||| and a tv for "
                            "rosario cabedo .",
                        ),
                        _row(3, "would you mind moving me to a quieter room ?"),
                        _row(4, "i have booked a room ."),
                        _row(5, "i think that there is a problem ."),
                        _row(6, "do you have any rooms with a tv , air conditioning and a safe available ?"),
                        _row(7, "would you mind showing us a room with a tv ?"),
                        _row(8, "does it have a telephone ?"),
                        _row(9, "i am leaving on the second at eight in the evening ."),
                        _row(10, "how much does a single room cost per week ?"),
                    ],
                )
            ]
        )

        alignment_corpus = DictionaryAlignmentCorpus(
            [
                MemoryAlignmentCollection(
                    "text1",
                    [
                        _alignment(1, AlignedWordPair(8, 9)),
                        _alignment(2, AlignedWordPair(6, 10)),
                        _alignment(3, AlignedWordPair(6, 8)),
                        _alignment(4, AlignedWordPair(6, 4)),
                        _alignment(5),
                        _alignment(6, AlignedWordPair(2, 4)),
                        _alignment(7, AlignedWordPair(5, 6)),
                        _alignment(8),
                        _alignment(9),
                        _alignment(10, AlignedWordPair(4, 5)),
                    ],
                )
            ]
        )

        corpus = source_corpus.align_rows(target_corpus, alignment_corpus)

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
        source_corpus = DictionaryTextCorpus([])
        target_corpus = DictionaryTextCorpus([])
        alignment_corpus = DictionaryAlignmentCorpus([])

        corpus = source_corpus.align_rows(target_corpus, alignment_corpus)

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


def _row(row_ref: int, text: str) -> TextRow:
    return TextRow("text1", row_ref, segment=[text])


def _alignment(row_ref: int, *pairs: AlignedWordPair) -> AlignmentRow:
    return AlignmentRow("text1", row_ref, aligned_word_pairs=pairs)

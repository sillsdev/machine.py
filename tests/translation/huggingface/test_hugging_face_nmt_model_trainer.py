import sys

if sys.platform == "darwin":
    from pytest import skip

    skip("skipping Hugging Face tests on MacOS", allow_module_level=True)

from tempfile import TemporaryDirectory

from transformers import Seq2SeqTrainingArguments

from machine.corpora import DictionaryTextCorpus, MemoryText, TextRow
from machine.translation.huggingface import HuggingFaceNmtEngine, HuggingFaceNmtModelTrainer


def test_train_non_empty_corpus() -> None:
    pretrained_engine = HuggingFaceNmtEngine("stas/tiny-m2m_100", src_lang="es", tgt_lang="en", max_length=20)
    pretrained_result = pretrained_engine.translate("una habitación individual por semana")

    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "¿Le importaría darnos las llaves de la habitación, por favor?"),
                        _row(
                            2,
                            "He hecho la reserva de una habitación tranquila doble con teléfono y televisión a nombre "
                            + "de Rosario Cabedo.",
                        ),
                        _row(3, "¿Le importaría cambiarme a otra habitación más tranquila?"),
                        _row(4, "Por favor, tengo reservada una habitación."),
                        _row(5, "Me parece que existe un problema."),
                        _row(6, "¿Tiene habitaciones libres con televisión, aire acondicionado y caja fuerte?"),
                        _row(7, "¿Le importaría mostrarnos una habitación con televisión?"),
                        _row(8, "¿Tiene teléfono?"),
                        _row(9, "Voy a marcharme el dos a las ocho de la noche."),
                        _row(10, "¿Cuánto cuesta una habitación individual por semana?"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Would you mind giving us the keys to the room, please?"),
                        _row(
                            2,
                            "I have made a reservation for a quiet, double room with a telephone and a TV for Rosario "
                            + "Cabedo.",
                        ),
                        _row(3, "Would you mind moving me to a quieter room?"),
                        _row(4, "I have booked a room."),
                        _row(5, "I think that there is a problem."),
                        _row(6, "Do you have any rooms with a TV, air conditioning and a safe available?"),
                        _row(7, "Would you mind showing us a room with a TV?"),
                        _row(8, "Does it have a telephone?"),
                        _row(9, "I am leaving on the second at eight in the evening."),
                        _row(10, "How much does a single room cost per week?"),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        trainer = HuggingFaceNmtModelTrainer(
            "stas/tiny-m2m_100",
            training_args,
            corpus,
            src_lang="es",
            tgt_lang="en",
            max_source_length=20,
            max_target_length=20,
        )
        trainer.train()
        trainer.save()

        finetuned_engine = HuggingFaceNmtEngine(temp_dir, src_lang="es", tgt_lang="en", max_length=20)
        finetuned_result = finetuned_engine.translate("una habitación individual por semana")
        assert finetuned_result.translation != pretrained_result.translation


def test_update_tokenizer_missing_char() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Ḻ ḻ Ṉ"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "ॽ " + "‌ " + "‍"),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        trainer_nochar = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
        )
        trainer_nochar.train()
        trainer_nochar.save()

        finetuned_engine_nochar = HuggingFaceNmtEngine(
            temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20
        )
        finetuned_result_nochar = finetuned_engine_nochar.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        config = {"tokenizer": {"update_src": True, "update_trg": True}}
        trainer_char = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
            config=config,
        )
        trainer_char.train()
        trainer_char.save()

        finetuned_engine_char = HuggingFaceNmtEngine(temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20)
        finetuned_result_char = finetuned_engine_char.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        assert finetuned_result_nochar.translation != finetuned_result_char.translation


def test_update_tokenizer_missing_char_src() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Ḻ ḻ Ṉ"),
                        _row(2, "ॽ " + "‌ " + "‍"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "random data"),
                        _row(2, "other info"),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        trainer_nochar = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
        )
        trainer_nochar.train()
        trainer_nochar.save()

        finetuned_engine_nochar = HuggingFaceNmtEngine(
            temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20
        )
        finetuned_result_nochar = finetuned_engine_nochar.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        config = {"tokenizer": {"update_src": True, "update_trg": False}}
        trainer_char = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
            config=config,
        )
        trainer_char.train()
        trainer_char.save()

        finetuned_engine_char = HuggingFaceNmtEngine(temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20)
        finetuned_result_char = finetuned_engine_char.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        assert finetuned_result_nochar.translation != finetuned_result_char.translation


def test_update_tokenizer_missing_char_trg() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "random data"),
                        _row(2, "other info"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Ḻ ḻ Ṉ"),
                        _row(2, "ॽ " + "‌ " + "‍"),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        trainer_nochar = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
        )
        trainer_nochar.train()
        trainer_nochar.save()

        finetuned_engine_nochar = HuggingFaceNmtEngine(
            temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20
        )
        finetuned_result_nochar = finetuned_engine_nochar.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        config = {"tokenizer": {"update_src": False, "update_trg": True}}
        trainer_char = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
            config=config,
        )
        trainer_char.train()
        trainer_char.save()

        finetuned_engine_char = HuggingFaceNmtEngine(temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20)
        finetuned_result_char = finetuned_engine_char.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        assert finetuned_result_nochar.translation != finetuned_result_char.translation


def test_update_tokenizer_no_missing_char() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "¿Le importaría darnos las llaves de la habitación, por favor?"),
                        _row(
                            2,
                            "He hecho la reserva de una habitación tranquila doble con teléfono y televisión a nombre "
                            + "de Rosario Cabedo.",
                        ),
                        _row(3, "¿Le importaría cambiarme a otra habitación más tranquila?"),
                        _row(4, "Por favor, tengo reservada una habitación."),
                        _row(5, "Me parece que existe un problema."),
                        _row(6, "¿Tiene habitaciones libres con televisión, aire acondicionado y caja fuerte?"),
                        _row(7, "¿Le importaría mostrarnos una habitación con televisión?"),
                        _row(8, "¿Tiene teléfono?"),
                        _row(9, "Voy a marcharme el dos a las ocho de la noche."),
                        _row(10, "¿Cuánto cuesta una habitación individual por semana?"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Would you mind giving us the keys to the room, please?"),
                        _row(
                            2,
                            "I have made a reservation for a quiet, double room with a telephone and a TV for Rosario "
                            + "Cabedo.",
                        ),
                        _row(3, "Would you mind moving me to a quieter room?"),
                        _row(4, "I have booked a room."),
                        _row(5, "I think that there is a problem."),
                        _row(6, "Do you have any rooms with a TV, air conditioning and a safe available?"),
                        _row(7, "Would you mind showing us a room with a TV?"),
                        _row(8, "Does it have a telephone?"),
                        _row(9, "I am leaving on the second at eight in the evening."),
                        _row(10, "How much does a single room cost per week?"),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        trainer_nochar = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
        )
        trainer_nochar.train()
        trainer_nochar.save()

        finetuned_engine_nochar = HuggingFaceNmtEngine(
            temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20
        )
        finetuned_result_nochar = finetuned_engine_nochar.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        config = {"tokenizer": {"update_src": True, "update_trg": True}}
        trainer_char = HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            max_source_length=20,
            max_target_length=20,
            config=config,
        )
        trainer_char.train()
        trainer_char.save()

        finetuned_engine_char = HuggingFaceNmtEngine(temp_dir, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=20)
        finetuned_result_char = finetuned_engine_char.translate(
            "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
        )

        assert finetuned_result_nochar.translation == finetuned_result_char.translation


def _row(row_ref: int, text: str) -> TextRow:
    return TextRow("text1", row_ref, segment=[text])

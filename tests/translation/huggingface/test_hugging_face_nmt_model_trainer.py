import sys

if sys.platform == "darwin":
    from pytest import skip

    skip("skipping Hugging Face tests on MacOS", allow_module_level=True)

from tempfile import TemporaryDirectory
from typing import cast

from transformers import (
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
)

from machine.corpora import DictionaryTextCorpus, MemoryText, TextRow
from machine.translation.huggingface import HuggingFaceNmtEngine, HuggingFaceNmtModelTrainer, add_lang_code_to_tokenizer


def test_train_non_empty_corpus() -> None:
    with HuggingFaceNmtEngine("stas/tiny-m2m_100", src_lang="es", tgt_lang="en", max_length=20) as pretrained_engine:
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

        with HuggingFaceNmtModelTrainer(
            "stas/tiny-m2m_100",
            training_args,
            corpus,
            src_lang="es",
            tgt_lang="en",
            max_src_length=20,
            max_tgt_length=20,
        ) as trainer:
            trainer.train()
            trainer.save()

        with HuggingFaceNmtEngine(temp_dir, src_lang="es", tgt_lang="en", max_length=20) as finetuned_engine:
            finetuned_result = finetuned_engine.translate("una habitación individual por semana")

        assert finetuned_result.translation != pretrained_result.translation


def test_update_tokenizer_missing_char() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Ḏ Ḻ ḻ Ṉ"),
                        _row(2, "d e f g"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "a b c"),
                        _row(1, "ॽ " + "‌ " + "‍"),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=False,
            add_unk_tgt_tokens=False,
        ) as trainer_nochar:
            trainer_nochar.train()
            trainer_nochar.save()

        with HuggingFaceNmtEngine(
            temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20
        ) as finetuned_engine_nochar:
            assert isinstance(finetuned_engine_nochar.tokenizer, PreTrainedTokenizerFast)
            finetuned_result_nochar = finetuned_engine_nochar.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )
            finetuned_result_nochar_composite = finetuned_engine_nochar.tokenizer.encode("Ḏ is a composite character")
            norm_result_nochar1 = finetuned_engine_nochar.tokenizer.backend_tokenizer.normalizer.normalize_str("‌ ")
            norm_result_nochar2 = finetuned_engine_nochar.tokenizer.backend_tokenizer.normalizer.normalize_str("‍")

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=True,
            add_unk_tgt_tokens=True,
        ) as trainer_char:
            trainer_char.train()
            trainer_char.save()

        with HuggingFaceNmtEngine(temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20) as finetuned_engine_char:
            assert isinstance(finetuned_engine_char.tokenizer, PreTrainedTokenizerFast)
            finetuned_result_char = finetuned_engine_char.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )
            finetuned_result_char_composite = finetuned_engine_char.tokenizer.encode("Ḏ is a composite character")
            norm_result_char1 = finetuned_engine_char.tokenizer.backend_tokenizer.normalizer.normalize_str("‌ ")
            norm_result_char2 = finetuned_engine_char.tokenizer.backend_tokenizer.normalizer.normalize_str("‍")

        assert norm_result_nochar1 != norm_result_char1
        assert norm_result_nochar2 != norm_result_char2

        assert finetuned_result_nochar != finetuned_result_char
        assert finetuned_result_nochar_composite != finetuned_result_char_composite


def test_update_tokenizer_missing_char_skip() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Ḻ ḻ Ṉ"),
                        _row(2, "d e f"),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "a b c"),
                        _row(2, "ॽ " + "‌ " + "‍"),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        with HuggingFaceNmtModelTrainer(
            "stas/tiny-m2m_100",
            training_args,
            corpus,
            src_lang="en",
            tgt_lang="es",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=False,
            add_unk_tgt_tokens=False,
        ) as trainer_nochar:
            trainer_nochar.train()
            trainer_nochar.save()

        with HuggingFaceNmtEngine(temp_dir, src_lang="en", tgt_lang="es", max_length=20) as finetuned_engine_nochar:
            finetuned_result_nochar = finetuned_engine_nochar.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )

        with HuggingFaceNmtModelTrainer(
            "stas/tiny-m2m_100",
            training_args,
            corpus,
            src_lang="en",
            tgt_lang="es",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=False,
            add_unk_tgt_tokens=False,
        ) as trainer_char:
            trainer_char.train()
            trainer_char.save()

        with HuggingFaceNmtEngine(temp_dir, src_lang="en", tgt_lang="es", max_length=20) as finetuned_engine_char:
            finetuned_result_char = finetuned_engine_char.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )

        assert finetuned_result_nochar == finetuned_result_char


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

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=False,
            add_unk_tgt_tokens=False,
        ) as trainer_nochar:
            trainer_nochar.train()
            trainer_nochar.save()

        with HuggingFaceNmtEngine(
            temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20
        ) as finetuned_engine_nochar:
            finetuned_result_nochar = finetuned_engine_nochar.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=True,
            add_unk_tgt_tokens=False,
        ) as trainer_char:
            trainer_char.train()
            trainer_char.save()

        with HuggingFaceNmtEngine(temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20) as finetuned_engine_char:
            finetuned_result_char = finetuned_engine_char.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )

        assert finetuned_result_nochar != finetuned_result_char


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

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=False,
            add_unk_tgt_tokens=False,
        ) as trainer_nochar:
            trainer_nochar.train()
            trainer_nochar.save()

        with HuggingFaceNmtEngine(
            temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20
        ) as finetuned_engine_nochar:
            finetuned_result_nochar = finetuned_engine_nochar.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=False,
            add_unk_tgt_tokens=True,
        ) as trainer_char:
            trainer_char.train()
            trainer_char.save()

        with HuggingFaceNmtEngine(temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20) as finetuned_engine_char:
            finetuned_result_char = finetuned_engine_char.tokenizer.encode(
                "Ḻ, ḻ, Ṉ, ॽ, " + "‌  and " + "‍" + " are new characters"
            )

        assert finetuned_result_nochar != finetuned_result_char


def test_update_tokenizer_no_missing_char() -> None:
    with TemporaryDirectory() as temp_dir:
        source_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "Me parece que existe un problema."),
                        _row(2, "Yo voy a marcharme el dos a las ocho de la noche."),
                    ],
                )
            ]
        )

        target_corpus = DictionaryTextCorpus(
            [
                MemoryText(
                    "text1",
                    [
                        _row(1, "I think that there is a problem."),
                        _row(2, "I am leaving on the second at eight in the evening."),
                    ],
                )
            ]
        )
        corpus = source_corpus.align_rows(target_corpus)

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir, num_train_epochs=1, report_to=["none"], learning_rate=0.01
        )

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=False,
            add_unk_tgt_tokens=False,
        ) as trainer_nochar:
            trainer_nochar.train()
            trainer_nochar.save()

        with HuggingFaceNmtEngine(
            temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20
        ) as finetuned_engine_nochar:
            finetuned_result_nochar = finetuned_engine_nochar.tokenizer.encode("una habitación individual por semana")

        with HuggingFaceNmtModelTrainer(
            "hf-internal-testing/tiny-random-nllb",
            training_args,
            corpus,
            src_lang="en_XX",
            tgt_lang="es_XX",
            max_src_length=20,
            max_tgt_length=20,
            add_unk_src_tokens=True,
            add_unk_tgt_tokens=True,
        ) as trainer_char:
            trainer_char.train()
            trainer_char.save()

        with HuggingFaceNmtEngine(temp_dir, src_lang="en_XX", tgt_lang="es_XX", max_length=20) as finetuned_engine_char:
            finetuned_result_char = finetuned_engine_char.tokenizer.encode("una habitación individual por semana")

        assert finetuned_result_nochar == finetuned_result_char


def test_nllb_tokenizer_add_lang_code() -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = cast(NllbTokenizer, NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"))
        assert "new_lang" not in tokenizer.added_tokens_encoder
        add_lang_code_to_tokenizer(tokenizer, "new_lang")
        assert "new_lang" in tokenizer.added_tokens_encoder
        tokenizer.save_pretrained(temp_dir)
        new_tokenizer = cast(NllbTokenizer, NllbTokenizer.from_pretrained(temp_dir))
        assert "new_lang" in new_tokenizer.added_tokens_encoder
    return


def test_nllb_tokenizer_fast_add_lang_code() -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = cast(NllbTokenizerFast, NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M"))
        assert "new_lang" not in tokenizer.added_tokens_encoder
        add_lang_code_to_tokenizer(tokenizer, "new_lang")
        assert "new_lang" in tokenizer.added_tokens_encoder
        tokenizer.save_pretrained(temp_dir)
        new_tokenizer = cast(NllbTokenizerFast, NllbTokenizerFast.from_pretrained(temp_dir))
        assert "new_lang" in new_tokenizer.added_tokens_encoder
    return


def test_mbart_tokenizer_add_lang_code() -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = cast(MBartTokenizer, MBartTokenizer.from_pretrained("hf-internal-testing/tiny-random-nllb"))
        assert "nl_NS" not in tokenizer.added_tokens_encoder
        add_lang_code_to_tokenizer(tokenizer, "nl_NS")
        assert "nl_NS" in tokenizer.added_tokens_encoder
        tokenizer.save_pretrained(temp_dir)
        new_tokenizer = cast(MBartTokenizer, MBartTokenizer.from_pretrained(temp_dir))
        assert "nl_NS" in new_tokenizer.added_tokens_encoder
    return


def test_mbart_tokenizer_fast_add_lang_code() -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = cast(MBartTokenizerFast, MBartTokenizerFast.from_pretrained("hf-internal-testing/tiny-random-nllb"))
        assert "nl_NS" not in tokenizer.added_tokens_encoder
        add_lang_code_to_tokenizer(tokenizer, "nl_NS")
        assert "nl_NS" in tokenizer.added_tokens_encoder
        tokenizer.save_pretrained(temp_dir)
        new_tokenizer = cast(MBartTokenizerFast, MBartTokenizerFast.from_pretrained(temp_dir))
        assert "nl_NS" in new_tokenizer.added_tokens_encoder
    return


def test_mbart_50_tokenizer_add_lang_code() -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = cast(MBart50Tokenizer, MBart50Tokenizer.from_pretrained("hf-internal-testing/tiny-random-mbart50"))
        assert "nl_NS" not in tokenizer.added_tokens_encoder
        add_lang_code_to_tokenizer(tokenizer, "nl_NS")
        assert "nl_NS" in tokenizer.added_tokens_encoder
        tokenizer.save_pretrained(temp_dir)
        new_tokenizer = cast(MBart50Tokenizer, MBart50Tokenizer.from_pretrained(temp_dir))
        assert "nl_NS" in new_tokenizer.added_tokens_encoder
    return


def test_mbart_50_tokenizer_fast_add_lang_code() -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = cast(
            MBart50TokenizerFast, MBart50TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-mbart50")
        )
        assert "nl_NS" not in tokenizer.added_tokens_encoder
        add_lang_code_to_tokenizer(tokenizer, "nl_NS")
        assert "nl_NS" in tokenizer.added_tokens_encoder
        tokenizer.save_pretrained(temp_dir)
        new_tokenizer = cast(MBart50TokenizerFast, MBart50TokenizerFast.from_pretrained(temp_dir))
        assert "nl_NS" in new_tokenizer.added_tokens_encoder
    return


def test_m2m_100_tokenizer_add_lang_code() -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = cast(M2M100Tokenizer, M2M100Tokenizer.from_pretrained("stas/tiny-m2m_100"))
        assert "nc" not in tokenizer.lang_code_to_id
        assert "__nc__" not in tokenizer.added_tokens_encoder
        add_lang_code_to_tokenizer(tokenizer, "nc")
        assert "nc" in tokenizer.lang_code_to_id
        assert "__nc__" in tokenizer.added_tokens_encoder
        tokenizer.save_pretrained(temp_dir)
        new_tokenizer = cast(M2M100Tokenizer, M2M100Tokenizer.from_pretrained(temp_dir))
        assert "nc" in tokenizer.lang_code_to_id
        assert "__nc__" in new_tokenizer.added_tokens_encoder
    return


def _row(row_ref: int, text: str) -> TextRow:
    return TextRow("text1", row_ref, segment=[text])

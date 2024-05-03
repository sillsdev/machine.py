from typing import Any, Iterable

from machine.corpora.memory_text import MemoryText
from machine.corpora.text import Text
from machine.corpora.text_corpus import TextCorpus
from machine.corpora.text_row import TextRow, TextRowFlags
from machine.translation.unigram_truecaser import UnigramTruecaser
from machine.translation.unigram_truecaser_trainer import UnigramTruecaserTrainer

training_segments = [
    ["The", "house", "is", "made", "of", "wood", "."],
    ["I", "go", "on", "adventures", "."],
    ["He", "read", "the", "book", "about", "Sherlock", "Holmes", "."],
    ["John", "and", "I", "agree", "that", "you", "and", "I", "are", "smart", "."],
]


def test_truecase_empty() -> None:
    truecaser = create_truecaser()
    result = truecaser.truecase([])
    assert result == []


def test_truecase_capitialized_name() -> None:
    truecaser = create_truecaser()
    result = truecaser.truecase(["THE", "ADVENTURES", "OF", "SHERLOCK", "HOLMES"])
    assert result == ["the", "adventures", "of", "Sherlock", "Holmes"]


def test_truecase_unknown_word() -> None:
    truecaser = create_truecaser()
    result = truecaser.truecase(["THE", "EXPLOITS", "OF", "SHERLOCK", "HOLMES"])
    assert result == ["the", "EXPLOITS", "of", "Sherlock", "Holmes"]


def test_truecase_multiple_sentences() -> None:
    truecaser = create_truecaser()
    result = truecaser.truecase(["SHERLOCK", "HOLMES", "IS", "SMART", ".", "YOU", "AGREE", "."])
    assert result == ["Sherlock", "Holmes", "is", "smart", ".", "you", "agree", "."]


def test_truecase_ignore_first_word_during_training() -> None:
    truecaser = create_truecaser()
    result = truecaser.truecase(["HE", "IS", "SMART", "."])
    assert result == ["HE", "is", "smart", "."]


def create_truecaser() -> UnigramTruecaser:
    truecaser = UnigramTruecaser()
    for segment in training_segments:
        truecaser.train_segment(segment)
    return truecaser


class MemoryTextCorpus(TextCorpus):
    def __init__(self, id: str, rows: Iterable[TextRow] = []) -> None:
        self._id = id
        self._rows = list(rows)

    @property
    def texts(self) -> Iterable[Text]:
        return [MemoryText(self._id, self._rows)]

    @property
    def is_tokenized(self) -> bool:
        return True


def text_row(text_id: str, ref: Any, text: str = "", flags: TextRowFlags = TextRowFlags.SENTENCE_START) -> TextRow:
    return TextRow(text_id, ref, [] if len(text) == 0 else text.split(), flags)


def test_compare_with_truecase_trainer() -> None:
    text = MemoryTextCorpus(
        "text1", [text_row("text1", i, " ".join(segment)) for i, segment in enumerate(training_segments)]
    )
    trainer = UnigramTruecaserTrainer(text)
    trainer.new_truecaser = UnigramTruecaser()
    trainer.train()

    truecaser = create_truecaser()

    assert trainer.new_truecaser._bestTokens == truecaser._bestTokens
    assert trainer.new_truecaser._casing == truecaser._casing

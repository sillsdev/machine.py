from machine.tokenization import ZwspWordTokenizer


def test_tokenize_empty() -> None:
    tokenizer = ZwspWordTokenizer()
    assert not any(tokenizer.tokenize(""))


def test_tokenize_zswp() -> None:
    tokenizer = ZwspWordTokenizer()
    assert not any(tokenizer.tokenize("\u200b"))


def test_tokenize_space() -> None:
    tokenizer = ZwspWordTokenizer()
    assert list(tokenizer.tokenize("គែស\u200bមាង់ អី\u200bនៃ\u200bជេង\u200bនារ\u200bត៝ល់\u200bព្វាន់។")) == [
        "គែស",
        "មាង់",
        " ",
        "អី",
        "នៃ",
        "ជេង",
        "នារ",
        "ត៝ល់",
        "ព្វាន់",
        "។",
    ]


def test_tokenize_guillemet() -> None:
    tokenizer = ZwspWordTokenizer()
    assert list(tokenizer.tokenize("ឞ្ក្នៃ\u200bរាញា «នារ» ជេសរី")) == ["ឞ្ក្នៃ", "រាញា", "«", "នារ", "»", "ជេសរី"]


def test_tokenize_punctuation() -> None:
    tokenizer = ZwspWordTokenizer()
    assert list(tokenizer.tokenize("ไป\u200bไหน\u200bมา? เขา\u200bถาม\u200bผม.")) == [
        "ไป",
        "ไหน",
        "มา",
        "?",
        "เขา",
        "ถาม",
        "ผม",
        ".",
    ]
    assert list(tokenizer.tokenize("ช้าง, ม้า, วัว, กระบือ")) == ["ช้าง", ",", "ม้า", ",", "วัว", ",", "กระบือ"]


def test_tokenize_punctuation_inside_word() -> None:
    tokenizer = ZwspWordTokenizer()
    assert list(tokenizer.tokenize("เริ่ม\u200bต้น\u200bที่ 7,999 บาท")) == [
        "เริ่ม",
        "ต้น",
        "ที่",
        " ",
        "7,999",
        " ",
        "บาท",
    ]


def test_tokenize_multiple_spaces() -> None:
    tokenizer = ZwspWordTokenizer()
    assert list(tokenizer.tokenize("គែស\u200bមាង់  អី\u200bនៃ\u200bជេង\u200bនារ\u200bត៝ល់\u200bព្វាន់។")) == [
        "គែស",
        "មាង់",
        "  ",
        "អី",
        "នៃ",
        "ជេង",
        "នារ",
        "ត៝ល់",
        "ព្វាន់",
        "។",
    ]
    assert list(tokenizer.tokenize("ไป\u200bไหน\u200bมา?  เขา\u200bถาม\u200bผม.")) == [
        "ไป",
        "ไหน",
        "มา",
        "?",
        "เขา",
        "ถาม",
        "ผม",
        ".",
    ]

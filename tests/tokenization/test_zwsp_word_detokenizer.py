from machine.tokenization import ZwspWordDetokenizer


def test_detokenize_empty() -> None:
    detokenizer = ZwspWordDetokenizer()
    assert detokenizer.detokenize([]) == ""


def test_detokenize_space() -> None:
    detokenizer = ZwspWordDetokenizer()
    assert (
        detokenizer.detokenize(["គែស", "មាង់", " ", "អី", "នៃ", "ជេង", "នារ", "ត៝ល់", "ព្វាន់", "។"])
        == "គែស\u200bមាង់ អី\u200bនៃ\u200bជេង\u200bនារ\u200bត៝ល់\u200bព្វាន់។"
    )


def test_detokenize_guillment() -> None:
    detokenizer = ZwspWordDetokenizer()
    assert detokenizer.detokenize(["ឞ្ក្នៃ", "រាញា", "«", "នារ", "»", "ជេសរី"]) == "ឞ្ក្នៃ\u200bរាញា «នារ» ជេសរី"


def test_detokenize_punctuation() -> None:
    detokenizer = ZwspWordDetokenizer()
    assert (
        detokenizer.detokenize(["ไป", "ไหน", "มา", "?", "เขา", "ถาม", "ผม", "."])
        == "ไป\u200bไหน\u200bมา? เขา\u200bถาม\u200bผม."
    )

    assert detokenizer.detokenize(["ช้าง", ",", "ม้า", ",", "วัว", ",", "กระบือ"]) == "ช้าง, ม้า, วัว, กระบือ"


def test_detokenize_punctuation_inside_word() -> None:
    detokenizer = ZwspWordDetokenizer()
    assert detokenizer.detokenize(["เริ่ม", "ต้น", "ที่", " ", "7,999", " ", "บาท"]) == "เริ่ม\u200bต้น\u200bที่ 7,999 บาท"


def test_detokenize_multiple_spaces() -> None:
    detokenizer = ZwspWordDetokenizer()
    assert (
        detokenizer.detokenize(["គែស", "មាង់", "  ", "អី", "នៃ", "ជេង", "នារ", "ត៝ល់", "ព្វាន់", "។"])
        == "គែស\u200bមាង់  អី\u200bនៃ\u200bជេង\u200bនារ\u200bត៝ល់\u200bព្វាន់។"
    )

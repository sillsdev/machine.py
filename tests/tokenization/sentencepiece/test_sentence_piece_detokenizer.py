from machine.tokenization.sentencepiece import SentencePieceDetokenizer


def test_detokenize() -> None:
    detokenizer = SentencePieceDetokenizer()
    sentence = detokenizer.detokenize(
        (
            "▁In ▁particular , ▁the ▁actress es ▁play ▁a ▁major ▁role ▁in ▁the ▁sometimes ▁rather ▁dubious"
            + " ▁staging ."
        ).split()
    )
    assert sentence == "In particular, the actresses play a major role in the sometimes rather dubious staging."


def test_detokenize_empty() -> None:
    detokenizer = SentencePieceDetokenizer()
    sentence = detokenizer.detokenize([])
    assert sentence == ""

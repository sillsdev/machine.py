import unicodedata
from typing import Sequence


def lowercase(tokens: Sequence[str]) -> Sequence[str]:
    return [t.lower() for t in tokens]


def escape_spaces(tokens: Sequence[str]) -> Sequence[str]:
    return [("<space>" if len(t) > 0 and t.isspace() else t) for t in tokens]


def unescape_spaces(tokens: Sequence[str]) -> Sequence[str]:
    return [(" " if t == "<space>" else t) for t in tokens]


def normalize(normalization_form: str, tokens: Sequence[str]) -> Sequence[str]:
    return [unicodedata.normalize(normalization_form, t) for t in tokens]


def nfc_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFC", tokens)


def nfd_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFD", tokens)


def nfkc_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFKC", tokens)


def nfkd_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFKD", tokens)

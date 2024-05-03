import unicodedata
from typing import Literal, Sequence


def lowercase(tokens: Sequence[str]) -> Sequence[str]:
    return [t.lower() for t in tokens]


def escape_spaces(tokens: Sequence[str]) -> Sequence[str]:
    return [("<space>" if len(t) > 0 and t.isspace() else t) for t in tokens]


def unescape_spaces(tokens: Sequence[str]) -> Sequence[str]:
    return [(" " if t == "<space>" else t) for t in tokens]


def _get_normalization_form(normalization_form: str) -> Literal["NFC", "NFD", "NFKC", "NFKD"]:
    if normalization_form == "NFC":
        return "NFC"
    if normalization_form == "NFD":
        return "NFD"
    if normalization_form == "NFKC":
        return "NFKC"
    if normalization_form == "NFKD":
        return "NFKD"
    raise ValueError(f"Unknown normalization form: {normalization_form}")


def normalize(normalization_form: str, tokens: Sequence[str]) -> Sequence[str]:
    return [unicodedata.normalize(_get_normalization_form(normalization_form), t) for t in tokens]


def nfc_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFC", tokens)


def nfd_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFD", tokens)


def nfkc_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFKC", tokens)


def nfkd_normalize(tokens: Sequence[str]) -> Sequence[str]:
    return normalize("NFKD", tokens)

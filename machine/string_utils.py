import unicodedata

SENTENCE_TERMINALS = {
    ".",
    "!",
    "?",
    "\u203C",
    "\u203D",
    "\u2047",
    "\u2048",
    "\u2049",
    "\u3002",
    "\uFE52",
    "\uFE57",
    "\uFF01",
    "\uFF0E",
    "\uFF1F",
    "\uFF61",
}

QUOTATION_MARKS = {'"', "“", "”", "„", "‟", "'", "‘", "’", "‚", "‛", "«", "»", "‹", "›"}

DELAYED_SENTENCE_END = QUOTATION_MARKS | {")", "]", ">", "}"}


def is_sentence_terminal(s: str) -> bool:
    return len(s) > 0 and all(c in SENTENCE_TERMINALS for c in s)


def is_delayed_sentence_end(s: str) -> bool:
    return len(s) > 0 and all(c in DELAYED_SENTENCE_END for c in s)


def is_punctuation(c: str) -> bool:
    category = unicodedata.category(c)
    return category.startswith("P")


def is_symbol(c: str) -> bool:
    category = unicodedata.category(c)
    return category.startswith("S")


def is_control(c: str) -> bool:
    category = unicodedata.category(c)
    return category == "Cc"

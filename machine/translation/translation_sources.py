from enum import Flag, auto


class TranslationSources(Flag):
    NONE = 0
    SMT = auto()
    TRANSFER = auto()
    PREFIX = auto()
    NMT = auto()

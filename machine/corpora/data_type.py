from enum import Enum, auto


class DataType(Enum):  # TODO what options to include? Does a verse=SENTENCE for our purposes?
    GLOSS = auto()
    PHRASE = auto()
    SENTENCE = auto()
    PASSAGE = auto()
    DOCUMENT = auto()

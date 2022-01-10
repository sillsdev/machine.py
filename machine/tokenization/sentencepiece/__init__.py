try:
    import sentencepiece  # noqa: F401
except ImportError:
    raise RuntimeError(
        'sil-machine must be installed with the "sentencepiece" extra in order to use the '
        + "machine.tokenization.sentencepiece package."
    )

from .sentence_piece_detokenizer import SentencePieceDetokenizer
from .sentence_piece_tokenizer import SentencePieceTokenizer

__all__ = ["SentencePieceDetokenizer", "SentencePieceTokenizer"]

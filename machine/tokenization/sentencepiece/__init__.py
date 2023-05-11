from ...utils.packages import is_sentencepiece_available

if not is_sentencepiece_available():
    raise RuntimeError('sentencepiece is not installed. Install sil-machine with the "sentencepiece" extra.')

from .sentence_piece_detokenizer import SentencePieceDetokenizer
from .sentence_piece_tokenizer import SentencePieceTokenizer

__all__ = ["SentencePieceDetokenizer", "SentencePieceTokenizer"]

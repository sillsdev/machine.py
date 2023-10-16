from ..tokenization.detokenizer import Detokenizer
from ..tokenization.range_tokenizer import RangeTokenizer
from ..tokenization.whitespace_detokenizer import WHITESPACE_DETOKENIZER
from ..tokenization.whitespace_tokenizer import WHITESPACE_TOKENIZER
from .error_correction_model import ErrorCorrectionModel
from .interactive_translation_engine import InteractiveTranslationEngine
from .interactive_translator import InteractiveTranslator


class InteractiveTranslatorFactory:
    def __init__(
        self,
        engine: InteractiveTranslationEngine,
        target_tokenizer: RangeTokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        target_detokenizer: Detokenizer[str, str] = WHITESPACE_DETOKENIZER,
    ) -> None:
        self._engine = engine
        self._ecm = ErrorCorrectionModel()
        self.target_tokenizer = target_tokenizer
        self.target_detokenizer = target_detokenizer

    @property
    def engine(self) -> InteractiveTranslationEngine:
        return self._engine

    @property
    def error_correction_model(self) -> ErrorCorrectionModel:
        return self._ecm

    def create(self, segment: str, sentence_start: bool = True) -> InteractiveTranslator:
        return InteractiveTranslator(
            self._ecm,
            self._engine,
            self.target_tokenizer,
            self.target_detokenizer,
            segment,
            self._engine.get_word_graph(segment),
            sentence_start,
        )

import os
import shutil
from pathlib import Path
from typing import Any

from ...tokenization.detokenizer import Detokenizer
from ...tokenization.latin_word_detokenizer import LatinWordDetokenizer
from ...tokenization.latin_word_tokenizer import LatinWordTokenizer
from ...tokenization.tokenizer import Tokenizer
from ...tokenization.whitespace_detokenizer import WhitespaceDetokenizer
from ...tokenization.whitespace_tokenizer import WhitespaceTokenizer
from ...tokenization.zwsp_word_detokenizer import ZwspWordDetokenizer
from ...tokenization.zwsp_word_tokenizer import ZwspWordTokenizer

_THOT_NEW_MODEL_DIRECTORY = os.path.join(os.path.dirname(__file__), "thot-new-model")

_TOKENIZERS = ["latin", "whitespace", "zwsp"]


class ThotModelFactory:
    def __init__(self, config: Any) -> None:
        self._config = config

    def init(self) -> None:
        shutil.copytree(_THOT_NEW_MODEL_DIRECTORY, self._model_dir, dirs_exist_ok=True)

    def create_tokenizer(self) -> Tokenizer[str, int, str]:
        name: str = self._config.thot.tokenizer
        name = name.lower()
        if name == "latin":
            return LatinWordTokenizer()
        if name == "whitespace":
            return WhitespaceTokenizer()
        if name == "zwsp":
            return ZwspWordTokenizer()
        raise RuntimeError(f"Unknown tokenizer: {name}.  Available tokenizers are: {_TOKENIZERS}.")

    def create_detokenizer(self) -> Detokenizer[str, str]:
        name: str = self._config.thot.tokenizer
        name = name.lower()
        if name == "latin":
            return LatinWordDetokenizer()
        if name == "whitespace":
            return WhitespaceDetokenizer()
        if name == "zwsp":
            return ZwspWordDetokenizer()
        raise RuntimeError(f"Unknown detokenizer: {name}.  Available detokenizers are: {_TOKENIZERS}.")

    def save_model(self) -> Path:
        tar_file_basename = Path(
            self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model"
        )
        return Path(shutil.make_archive(str(tar_file_basename), "gztar", self._model_dir))

    @property
    def _model_dir(self) -> Path:
        return Path(self._config.data_dir, self._config.shared_file_folder, "builds", self._config.build_id, "model")

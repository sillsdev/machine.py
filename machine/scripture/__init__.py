from typing import Any

from .canon import *
from .verse_ref import *
from .versification import *

ORIGINAL_VERSIFICATION: Versification
ENGLISH_VERSIFICATION: Versification
SEPTUAGINT_VERSIFICATION: Versification
VULGATE_VERSIFICATION: Versification
RUSSIAN_ORTHODOX_VERSIFICATION: Versification
RUSSIAN_PROTESTANT_VERSIFICATION: Versification


def __getattr__(name: str) -> Any:
    if name.endswith("_VERSIFICATION"):
        index = name.rindex("_")
        return Versification.get_builtin(name[:index])
    raise AttributeError

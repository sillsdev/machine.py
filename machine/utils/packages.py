from importlib.util import find_spec


def is_torch_available() -> bool:
    return find_spec("torch") is not None


def is_transformers_available() -> bool:
    return find_spec("transformers") is not None


def is_thot_available() -> bool:
    return find_spec("thot") is not None


def is_opennmt_available() -> bool:
    return find_spec("opennmt") is not None


def is_tensorflow_available() -> bool:
    return find_spec("tensorflow") is not None


def is_sentencepiece_available() -> bool:
    return find_spec("sentencepiece") is not None

from ...utils.packages import is_opennmt_available, is_tensorflow_available

if not is_opennmt_available():
    raise RuntimeError('opennmt-tf is not installed. Install sil-machine with the "opennmt" extra.')

if not is_tensorflow_available():
    raise RuntimeError("tensorflow is not installed.")

from .open_nmt_model_factory import OpenNmtModelFactory

__all__ = ["OpenNmtModelFactory"]

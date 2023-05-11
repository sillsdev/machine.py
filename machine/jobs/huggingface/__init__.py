from ...utils.packages import is_torch_available, is_transformers_available

if not is_transformers_available():
    raise RuntimeError('transformers is not installed. Install sil-machine with the "huggingface" extra.')

if not is_torch_available():
    raise RuntimeError("torch is not installed.")

from .hugging_face_nmt_model_factory import HuggingFaceNmtModelFactory

__all__ = ["HuggingFaceNmtModelFactory"]

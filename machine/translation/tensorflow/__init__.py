try:
    import tensorflow  # noqa: F401
except ImportError:
    raise RuntimeError(
        'sil-machine must be installed with the "tensorflow" extra in order to use the machine.translation.tensorflow'
        + " package."
    )

from .saved_model_nmt_engine import SavedModelNmtEngine

__all__ = ["SavedModelNmtEngine"]

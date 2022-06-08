from .canceled_error import CanceledError
from .context_managed_generator import ContextManagedGenerator
from .phased_progress_reporter import Phase, PhasedProgressReporter, PhaseProgress
from .progress_status import ProgressStatus

__all__ = [
    "CanceledError",
    "ContextManagedGenerator",
    "Phase",
    "PhasedProgressReporter",
    "PhaseProgress",
    "ProgressStatus",
]


def merge_dict(dict1: dict, dict2: dict, override_keys=None) -> dict:
    """Merges :obj:`dict2` into :obj:`dict1`.

    Args:
      dict1: The base dictionary.
      dict2: The dictionary to merge.
      override_keys: The values associated with these keys are overridden instead
        of merged.

    Returns:
      The merged dictionary :obj:`dict1`.
    """
    for key, value in dict2.items():
        if isinstance(value, dict) and (override_keys is None or key not in override_keys):
            dict1[key] = merge_dict(dict1.get(key, {}), value, override_keys=override_keys)
        else:
            dict1[key] = value
    return dict1

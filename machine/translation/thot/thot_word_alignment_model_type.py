from enum import Enum, auto

import thot.alignment as ta


class ThotWordAlignmentModelType(Enum):
    IBM1 = auto()
    IBM2 = auto()
    IBM3 = auto()
    IBM4 = auto()
    HMM = auto()
    FAST_ALIGN = auto()


def create_alignment_model(model_type: ThotWordAlignmentModelType) -> ta.AlignmentModel:
    if model_type is ThotWordAlignmentModelType.IBM1:
        return ta.Ibm1AlignmentModel()
    elif model_type is ThotWordAlignmentModelType.IBM2:
        return ta.Ibm2AlignmentModel()
    elif model_type is ThotWordAlignmentModelType.IBM3:
        return ta.Ibm3AlignmentModel()
    elif model_type is ThotWordAlignmentModelType.IBM4:
        return ta.Ibm4AlignmentModel()
    elif model_type is ThotWordAlignmentModelType.HMM:
        return ta.HmmAlignmentModel()
    elif model_type is ThotWordAlignmentModelType.FAST_ALIGN:
        return ta.FastAlignModel()
    else:
        raise ValueError("The model type is invalid.")

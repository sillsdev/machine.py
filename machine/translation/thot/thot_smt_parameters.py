from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Sequence, Tuple

from ...utils.typeshed import StrPath


class ModelHeuristic(IntEnum):
    NO_HEURISTIC = 0
    LOCAL_T = 4
    LOCAL_TD = 6


class LearningAlgorithm(IntEnum):
    BASIC_INCREMENTAL_TRAINING = 0
    MINIBATCH_TRAINING = 1
    BATCH_RETRAINING = 2


class LearningRatePolicy(IntEnum):
    FIXED = 0
    LIANG = 1
    OWN = 2
    WER_BASED = 3


def get_thot_smt_parameter(line: str) -> Tuple[str, str]:
    line = line.strip()
    if line.startswith("#"):
        return "", ""

    index = line.find(" ")
    if index == -1:
        name = line
        value = ""
    else:
        name = line[:index]
        value = line[index + 1 :].strip()

    if name.startswith("-"):
        name = name[1:]
    return name, value


@dataclass
class ThotSmtParameters:
    @classmethod
    def load(cls, config_filename: StrPath) -> ThotSmtParameters:
        parameters = ThotSmtParameters()
        config_dir = os.path.dirname(config_filename)

        with open(config_filename, "r", encoding="utf-8") as file:
            for line in file:
                name, value = get_thot_smt_parameter(line)
                if name == "tm":
                    if value == "":
                        raise ValueError("The -tm parameter does not have a value.")
                    parameters.translation_model_filename_prefix = value
                    if not os.path.isabs(parameters.translation_model_filename_prefix) and config_dir != "":
                        parameters.translation_model_filename_prefix = os.path.join(
                            config_dir, parameters.translation_model_filename_prefix
                        )
                elif name == "lm":
                    if value == "":
                        raise ValueError("The -lm parameter does not have a value.")
                    parameters.language_model_filename_prefix = value
                    if not os.path.isabs(parameters.language_model_filename_prefix) and config_dir != "":
                        parameters.language_model_filename_prefix = os.path.join(
                            config_dir, parameters.language_model_filename_prefix
                        )
                elif name == "W":
                    if value == "":
                        raise ValueError("The -W parameter does not have a value.")
                    parameters.model_w = float(value)
                elif name == "S":
                    if value == "":
                        raise ValueError("The -S parameter does not have a value.")
                    parameters.decoder_s = int(value)
                elif name == "A":
                    if value == "":
                        raise ValueError("The -A parameter does not have a value.")
                    parameters.model_a = int(value)
                elif name == "E":
                    if value == "":
                        raise ValueError("The -E parameter does not have a value.")
                    parameters.model_e = int(value)
                elif name == "nomon":
                    if value == "":
                        raise ValueError("The -nomon parameter does not have a value.")
                    parameters.model_non_monotonicity = int(value)
                elif name == "be":
                    parameters.decoder_breadth_first = False
                elif name == "G":
                    if value == "":
                        raise ValueError("The -G parameter does not have a value.")
                    parameters.decoder_g = int(value)
                elif name == "h":
                    if value == "":
                        raise ValueError("The -h parameter does not have a value.")
                    parameters.model_heuristic = ModelHeuristic(int(value))
                elif name == "olp":
                    if value == "":
                        raise ValueError("The -olp parameter does not have a value.")
                    tokens = value.split()
                    if len(tokens) >= 1:
                        parameters.learning_algorithm = LearningAlgorithm(int(tokens[0]))
                    if len(tokens) >= 2:
                        parameters.learning_rate_policy = LearningRatePolicy(int(tokens[1]))
                    if len(tokens) >= 3:
                        parameters.learning_step_size = float(tokens[2])
                    if len(tokens) >= 4:
                        parameters.learning_em_iters = int(tokens[3])
                    if len(tokens) >= 5:
                        parameters.learning_e = int(tokens[4])
                    if len(tokens) >= 6:
                        parameters.learning_r = int(tokens[5])
                elif name == "tmw":
                    if value == "":
                        raise ValueError("The -tmw parameter does not have a value.")
                    parameters.model_weights = [float(w) for w in value.split()]

        return parameters

    translation_model_filename_prefix: str = ""
    language_model_filename_prefix: str = ""
    model_non_monotonicity: int = 0
    model_w: float = 0.4
    model_a: int = 10
    model_e: int = 2
    model_heuristic: ModelHeuristic = ModelHeuristic.LOCAL_TD
    model_weights: Sequence[float] = field(default_factory=list)
    learning_algorithm: LearningAlgorithm = LearningAlgorithm.BASIC_INCREMENTAL_TRAINING
    learning_rate_policy: LearningRatePolicy = LearningRatePolicy.FIXED
    learning_step_size: float = 1
    learning_em_iters: int = 5
    learning_e: int = 1
    learning_r: int = 0
    decoder_s: int = 10
    decoder_breadth_first: bool = True
    decoder_g: int = 0

    def copy(self) -> ThotSmtParameters:
        return replace(self)

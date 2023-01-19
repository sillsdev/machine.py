from math import exp, log

LOG_SPACE_ONE = 0
LOG_SPACE_ZERO = -999999999


def log_space_add(logx: float, logy: float) -> float:
    if logx > logy:
        return logx + log(1 + exp(logy - logx))
    return logy + log(1 + exp(logx - logy))


def log_space_multiple(logx: float, logy: float) -> float:
    result = logx + logy
    if result < LOG_SPACE_ZERO:
        result = LOG_SPACE_ZERO
    return result


def log_space_divide(logx: float, logy: float) -> float:
    result = logx - logy
    if result < LOG_SPACE_ZERO:
        result = LOG_SPACE_ZERO
    return result


def to_log_space(value: float) -> float:
    if value == 0:
        return LOG_SPACE_ZERO
    return log(value)


def to_std_space(log_value: float) -> float:
    return exp(log_value)

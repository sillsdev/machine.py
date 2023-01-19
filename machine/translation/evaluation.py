from math import exp, log
from typing import Iterable, Sequence, Set, Tuple

_BLEU_N = 4


def compute_bleu(translations: Iterable[Sequence[str]], references: Iterable[Sequence[str]]) -> float:
    precs = [0.0] * _BLEU_N
    total = [0.0] * _BLEU_N
    trans_word_count = 0
    ref_word_count = 0

    for translation, reference in zip(translations, references):
        trans_word_count += len(translation)
        ref_word_count += len(reference)
        for n in range(1, _BLEU_N + 1):
            seg_prec, seg_total = _compute_bleu_precision(translation, reference, n)
            precs[n - 1] += seg_prec
            total[n - 1] += seg_total

    brevity_penalty = exp(1.0 - (ref_word_count / trans_word_count)) if trans_word_count < ref_word_count else 1.0

    bleu = 0.0
    bleus = [0.0] * _BLEU_N
    for n in range(1, _BLEU_N + 1):
        bleus[n - 1] = 0 if total[n - 1] == 0 else precs[n - 1] / total[n - 1]
        bleu += (1.0 / _BLEU_N) * (-999999999 if bleus[n - 1] == 0 else log(bleus[n - 1]))
    bleu = brevity_penalty * exp(bleu)
    return bleu


def _compute_bleu_precision(translation: Sequence[str], reference: Sequence[str], n: int) -> Tuple[int, int]:
    total = 0 if n > len(translation) else len(translation) - n + 1
    ref_total = 0 if n > len(reference) else len(reference) - n + 1

    matched: Set[int] = set()
    prec = 0
    for i in range(total):
        for j in range(ref_total):
            match = True
            for k in range(n):
                if translation[i + k] != reference[j + k]:
                    match = False
                    break

            if match and j not in matched:
                prec += 1
                matched.add(j)
                break
    return prec, total

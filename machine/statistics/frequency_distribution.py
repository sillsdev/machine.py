from typing import Dict, Iterable


class FrequencyDistribution:
    def __init__(self):
        self._sample_counts: Dict[str, int] = {}
        self.sample_outcome_count: int = 0

    def get_observed_samples(self) -> Iterable[str]:
        return self._sample_counts.keys()

    def increment(self, sample: str, count: int = 1) -> int:
        self._sample_counts[sample] = self._sample_counts.get(sample, 0) + count
        self.sample_outcome_count += count
        return self._sample_counts[sample]

    def decrement(self, sample: str, count: int = 1) -> int:
        if sample not in self._sample_counts:
            if count == 0:
                return 0
            else:
                raise ValueError(f'The sample "{sample}" cannot be decremented.')
        else:
            cur_count = self._sample_counts[sample]
            if count == 0:
                return cur_count
            if cur_count < count:
                raise ValueError(f'The sample "{sample}" cannot be decremented.')
            new_count = cur_count - count
            if new_count == 0:
                self._sample_counts.pop(sample)
            else:
                self._sample_counts[sample] = new_count
            self.sample_outcome_count -= count
            return new_count

    def __getitem__(self, item: str) -> int:
        if item not in self._sample_counts:
            self._sample_counts[item] = 0
        return self._sample_counts[item]

    def __iter__(self) -> Iterable[str]:
        return iter(self._sample_counts)

    def reset(self):
        self._sample_counts = {}

from typing import Collection, Dict, Iterable

from .frequency_distribution import FrequencyDistribution


class ConditionalFrequencyDistribution:

    def __init__(self):
        self._freq_dist: Dict[str, FrequencyDistribution] = {}

    def get_conditions(self) -> Collection[str]:
        return self._freq_dist.keys()

    def get_sample_outcome_count(self):
        return sum([fd.sample_outcome_count for fd in self._freq_dist.values()])

    def __getitem__(self, item: str) -> FrequencyDistribution:
        if item not in self._freq_dist:
            self._freq_dist[item] = FrequencyDistribution()
        return self._freq_dist[item]

    def __iter__(self) -> Iterable[str]:
        return iter(self._freq_dist)

    def reset(self):
        self._freq_dist = {}

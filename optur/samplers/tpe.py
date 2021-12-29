from typing import Dict, Optional, Sequence, List

from optur.proto.sampler_pb2 import SamplerConfig, RandomSamplerConfig
from optur.proto.search_space_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import Trial as TrialProto
from optur.samplers.random import RandomSampler
from optur.samplers.sampler import Sampler
from optur.utils.kernel_density_estimator.kde import KDE


class TPESampler(Sampler):
    def __init__(self, sampler_config: SamplerConfig) -> None:
        super().__init__(sampler_config=sampler_config)
        self._fallback_sampler = RandomSampler(SamplerConfig(random=RandomSamplerConfig()))
        self._kde: KDE
        self._upper_half_trials: List[TrialProto]
        self._lower_half_trials: List[TrialProto]

    def sync(self, trials: Sequence[TrialProto]) -> None:
        self._fallback_sampler.sync(trials)

    def joint_sample(
        self,
        fixed_parameters: Optional[Dict[str, ParameterValue]] = None,
        search_space: Optional[SearchSpace] = None,
    ) -> Dict[str, ParameterValue]:
        raise NotImplementedError()

    def sample(self, distribution: Distribution) -> ParameterValue:
        return self._fallback_sampler.sample(distribution=distribution)

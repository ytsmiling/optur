from typing import Dict, Optional, Sequence

from optur.proto.sampler_pb2 import RandomSamplerConfig, SamplerConfig
from optur.proto.search_space_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import Trial as TrialProto
from optur.samplers.random import RandomSampler
from optur.samplers.sampler import Sampler
from optur.utils.kernel_density_estimator.factorized_kde import FactorizedKDE
from optur.utils.kernel_density_estimator.kde import KDE
from optur.utils.sorted_trials import SortedTrials


class TPESampler(Sampler):
    def __init__(self, sampler_config: SamplerConfig) -> None:
        super().__init__(sampler_config=sampler_config)
        assert sampler_config.HasField("tpe")
        self._tpe_config = sampler_config.tpe
        self._fallback_sampler = RandomSampler(SamplerConfig(random=RandomSamplerConfig()))
        self._kde: KDE
        self._sorted_trials: SortedTrials

    def sync(self, trials: Sequence[TrialProto]) -> None:
        self._fallback_sampler.sync(trials)
        self._sorted_trials.sync(trials)

    def joint_sample(
        self,
        fixed_parameters: Optional[Dict[str, ParameterValue]] = None,
        search_space: Optional[SearchSpace] = None,
    ) -> Dict[str, ParameterValue]:
        sorted_trials = self._sorted_trials.to_list()
        if len(sorted_trials) < self._tpe_config.n_startup_trials:
            return {}
        # TODO(tsuzuku): Extend to MOTPE.
        half_idx = len(sorted_trials) // 2
        _less_half_trials = sorted_trials[:half_idx]  # D_l
        _greater_half_trials = sorted_trials[half_idx:]  # D_g
        assert _less_half_trials  # TODO(tsuzuku)
        assert _greater_half_trials  # TODO(tsuzuku)
        kde: KDE = FactorizedKDE()  # TODO(tsuzuku)
        if search_space is None:
            return {}
        kde.init(search_space=search_space, observations=[])  # TODO(tsuzuku)
        samples = kde.sample(fixed_parameters=fixed_parameters, k=self._tpe_config.n_ei_candidates)
        assert samples  # TODO(tsuzuku)
        # TODO(tsuzuku): Calculate log_probs and compare them.
        raise NotImplementedError()

    def sample(self, distribution: Distribution) -> ParameterValue:
        return self._fallback_sampler.sample(distribution=distribution)

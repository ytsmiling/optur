from typing import Dict, Optional, Sequence

from google.protobuf.timestamp_pb2 import Timestamp

from optur.errors import UnInitializedError
from optur.proto.sampler_pb2 import SamplerConfig
from optur.proto.study_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import Trial as TrialProto
from optur.samplers.backends.backend import SamplerBackend


class Sampler:
    def __init__(self, sampler_config: SamplerConfig) -> None:
        self._last_update_time = Timestamp(seconds=0, nanos=0)
        self._sampler_config = sampler_config
        self._backend: Optional[SamplerBackend] = None

    def init(self) -> None:
        pass

    @property
    def last_update_time(self) -> Timestamp:
        return self._last_update_time

    def update_timestamp(self, timestamp: Timestamp) -> None:
        self._last_update_time = timestamp

    def to_sampler_config(self) -> SamplerConfig:
        return self._sampler_config

    def sync(self, trials: Sequence[TrialProto]) -> None:
        """Update sampler-specific cache with the trials.

        Some trials might have already appeared in prior `sync` call.
        When trials appear multiple times with some finished state, whether
        samplers update their cache or not is sampler-dependent.
        When a trial appears for the first time after it finishes, the cache
        will be updated even if the trial haas appeared multiple times before the trial finishes.
        """

        if self._backend is None:
            raise UnInitializedError("Sampler is not initialized. Hint: call `Sampler.init()`.")
        self._backend.sync(trials=trials)

    def joint_sample(
        self,
        fixed_parameters: Optional[Dict[str, ParameterValue]] = None,
        search_space: Optional[SearchSpace] = None,
    ) -> Dict[str, ParameterValue]:
        """Perform joint-sampling for the trial.

        This method is expected to be called just after the trial starts running.
        This method accepts ``fixed_parameters`` because the trial object has non-empty parameters
        field if it was created by `add_trials` or `enqueue_trials`, and this can affects
        the result of the joint sampling.

        It is up to trials whether they use the result of the joint-sampling or not.
        """

        if self._backend is None:
            raise UnInitializedError("Sampler is not initialized. Hint: call `Sampler.init()`.")
        return self._backend.joint_sample(
            fixed_parameters=fixed_parameters, search_space=search_space
        )

    def sample(self, distribution: Distribution) -> ParameterValue:
        """Sample a parameter.

        This method will be called when ``trial.suggest_xxx` methods are called.
        """

        if self._backend is None:
            raise UnInitializedError("Sampler is not initialized. Hint: call `Sampler.init()`.")
        return self._backend.sample(distribution=distribution)

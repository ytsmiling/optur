from typing import Dict, Sequence

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.sampler_pb2 import SamplerConfig
from optur.proto.study_pb2 import Distribution, ParameterValue
from optur.proto.study_pb2 import Trial as TrialProto


class Sampler:
    def __init__(self) -> None:
        self._last_update_time = Timestamp(seconds=0, nanos=0)

    @property
    def last_update_time(self) -> Timestamp:
        return self._last_update_time

    def update_timestamp(self, timestamp: Timestamp) -> None:
        self._last_update_time = timestamp

    def to_sampler_config(self) -> SamplerConfig:
        pass

    def sync(self, trials: Sequence[TrialProto]) -> None:
        """Update sampler-specific cache with the trials.

        Some trials might have already appeared in prior `sync` call.
        When trials appear multiple times with some finished state, whether
        samplers update their cache or not is sampler-dependent.
        When a trial appears for the first time after it finishes, the cache
        will be updated even if the trial haas appeared multiple times before the trial finishes.
        """
        pass

    def joint_sample(self, trial: TrialProto) -> Dict[str, ParameterValue]:
        """Perform joint-sampling for the trial.

        This method is expected to be called just after the trial starts running.
        This method accepts `trial` object because the trial object has non-empty parameters
        field if it was created by `add_trials` or `enqueue_trials`, and this can affects
        the result of the sampling.

        It is up to trials whether they use the result of the joint-sampling or not.
        """
        pass

    def sample(self, distribution: Distribution) -> ParameterValue:
        """Sample parameters.

        This method will be called when ``trial.suggest_xxx` methods are called.
        """
        pass

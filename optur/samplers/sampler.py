import abc
from typing import Dict, NamedTuple, Optional, Sequence

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.sampler_pb2 import SamplerConfig
from optur.proto.search_space_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import AttributeValue
from optur.proto.study_pb2 import Trial as TrialProto


class JointSampleResult(NamedTuple):
    parameters: Dict[str, ParameterValue]
    system_attrs: Dict[str, AttributeValue]


class Sampler(abc.ABC):
    def __init__(self, sampler_config: SamplerConfig) -> None:
        self._sampler_config = sampler_config
        self._last_update_time: Optional[Timestamp] = None

    @property
    def last_update_time(self) -> Optional[Timestamp]:
        return self._last_update_time

    def update_timestamp(self, timestamp: Optional[Timestamp]) -> None:
        self._last_update_time = timestamp

    @abc.abstractclassmethod
    def set_search_space(self, search_space: Optional[SearchSpace]) -> None:
        pass

    def to_sampler_config(self) -> SamplerConfig:
        return self._sampler_config

    def sync(self, trials: Sequence[TrialProto]) -> None:
        """Update sampler-specific cache with the trials."""
        pass

    def joint_sample(
        self,
        fixed_parameters: Optional[Dict[str, ParameterValue]] = None,
    ) -> JointSampleResult:
        """Perform joint-sampling for the trial.

        When ``fixed_parameters`` is not :obj:`None`, the return value must includes them.

        This method is expected to be called just after the trial starts running.
        This method accepts ``fixed_parameters`` because the trial object has non-empty parameters
        field if it was created by `add_trials` or `enqueue_trials`, and this can affects
        the result of the joint sampling.

        This method might use an internal cache, but this method never updates internal
        states including the cache.

        It is up to trials whether they use the result of the joint-sampling or not.
        """

        return JointSampleResult(
            parameters=fixed_parameters.copy() if fixed_parameters is not None else {},
            system_attrs={},
        )

    @abc.abstractclassmethod
    def sample(self, distribution: Distribution) -> ParameterValue:
        """Sample a parameter.

        This method will be called when ``trial.suggest_xxx` methods are called.

        This method might use an internal cache, but this method never updates internal
        states including the cache.
        """
        pass

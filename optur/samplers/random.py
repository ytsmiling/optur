import random

from optur.proto.sampler_pb2 import RandomSamplerConfig, SamplerConfig
from optur.proto.study_pb2 import Distribution, ParameterValue
from optur.samplers.sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, config: RandomSamplerConfig) -> None:
        super().__init__(SamplerConfig(random=config))
        self._config = config

    def sample(self, distribution: Distribution) -> ParameterValue:
        """Sample a parameter."""
        if distribution.HasField("int_distribution"):
            # TODO(tsuzuku): Support log_scale.
            int_value = random.randint(
                distribution.int_distribution.low,
                distribution.int_distribution.high,
            )
            return ParameterValue(int_value=int_value)
        elif distribution.HasField("float_distribution"):
            # TODO(tsuzuku): Support log_scale.
            double_value = (
                distribution.float_distribution.low
                + (distribution.float_distribution.high - distribution.float_distribution.low)
                * random.random()
            )
            return ParameterValue(double_value=double_value)
        else:
            assert distribution.HasField("categorical_distribution")
            return random.choice(distribution.categorical_distribution.choices)

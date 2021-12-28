from optur.proto.sampler_pb2 import SamplerConfig
from optur.samplers.random import RandomSampler
from optur.samplers.sampler import Sampler


def create_sampler(sampler_config: SamplerConfig) -> Sampler:
    # TODO(tsuzuku): Implement this function.
    return RandomSampler(sampler_config.random)

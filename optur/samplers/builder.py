from optur.proto.sampler_pb2 import RandomSamplerConfig, SamplerConfig, TPESamplerConfig
from optur.samplers.random import RandomSampler
from optur.samplers.sampler import Sampler
from optur.samplers.tpe import TPESampler


def create_sampler(sampler_config: SamplerConfig) -> Sampler:
    if sampler_config.HasField("random"):
        return RandomSampler(sampler_config=sampler_config)
    if sampler_config.HasField("tpe"):
        return TPESampler(sampler_config=sampler_config)
    raise NotImplementedError("")  # TODO(tsuzuku)


def create_random_sampler() -> Sampler:
    return RandomSampler(sampler_config=SamplerConfig(random=RandomSamplerConfig()))


def create_tpe_sampler() -> Sampler:
    return TPESampler(
        sampler_config=SamplerConfig(
            tpe=TPESamplerConfig(
                n_startup_trials=20,
                n_ei_candidates=14,
            )
        )
    )

from optur.proto.sampler_pb2 import RandomSamplerConfig
from optur.proto.study_pb2 import Distribution, ParameterValue
from optur.samplers.backends.random import RandomSamplerBackend

FD = Distribution.FloatDistribution
ID = Distribution.IntDistribution
CD = Distribution.CategoricalDistribution


def test_random_sampler_backend_respects_float_distribution_ranges() -> None:
    sampler = RandomSamplerBackend(RandomSamplerConfig())
    for _ in range(10):
        distribution = Distribution(float_distribution=FD(low=10, high=100, log_scale=False))
        v = sampler.sample(distribution=distribution)
        if not v.HasField("double_value") or not 10 <= v.double_value <= 100:
            raise AssertionError(
                f"Random sampler was requested Distribution {distribution}, "
                f"but it returned {v}, which does not respect the distribution."
            )


def test_random_sampler_backend_respects_int_distribution_ranges() -> None:
    sampler = RandomSamplerBackend(RandomSamplerConfig())
    for _ in range(10):
        distribution = Distribution(int_distribution=ID(low=10, high=100, log_scale=False))
        v = sampler.sample(distribution=distribution)
        if not v.HasField("int_value") or not 10 <= v.int_value <= 100:
            raise AssertionError(
                f"Random sampler was requested Distribution {distribution}, "
                f"but it returned {v}, which does not respect the distribution."
            )


def test_random_sampler_backend_respects_categorical_distribution_choices() -> None:
    sampler = RandomSamplerBackend(RandomSamplerConfig())
    for _ in range(10):
        choices = [
            ParameterValue(string_value="foo"),
            ParameterValue(string_value="bar"),
            ParameterValue(int_value=46),
            ParameterValue(double_value=4.6),
        ]
        distribution = Distribution(categorical_distribution=CD(choices=choices))
        v = sampler.sample(distribution=distribution)
        if not any(v == c for c in choices):
            raise AssertionError(
                f"Random sampler was requested Distribution {distribution}, "
                f"but it returned {v}, which does not respect the distribution."
            )


# TODO(tsuzuku): Check empty categorical distribution handling.

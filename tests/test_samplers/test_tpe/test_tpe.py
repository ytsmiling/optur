import random

from optur.proto.sampler_pb2 import SamplerConfig, TPESamplerConfig
from optur.proto.search_space_pb2 import Distribution, ParameterValue
from optur.proto.study_pb2 import ObjectiveValue, Parameter, StudyInfo, Target, Trial
from optur.samplers.tpe import TPESampler


def int_distribution(low: int, high: int, log_scale: bool = False) -> Distribution:
    return Distribution(
        int_distribution=Distribution.IntDistribution(low=low, high=high, log_scale=log_scale)
    )


def float_distribution(low: float, high: float, log_scale: bool = False) -> Distribution:
    return Distribution(
        float_distribution=Distribution.FloatDistribution(low=low, high=high, log_scale=log_scale)
    )


def test_tpe_sampler_joint_sample_respects_fixed_parameters() -> None:
    # TODO(tsuzuku): Test this method.
    pass


def test_tpe_sampler_joint_sample_respects_int_range() -> None:
    sampler = TPESampler(
        sampler_config=SamplerConfig(
            tpe=TPESamplerConfig(n_ei_candidates=14),
        ),
        study_info=StudyInfo(
            targets=[Target(direction=Target.Direction.MINIMIZE)],
            # TODO(tsuzuku): Support search_space in study.
            # search_space=SearchSpace(
            #     distributions={
            #         "foo": int_distribution(low=2, high=12),
            #         "bar": float_distribution(low=-2.3, high=3.4),
            #     }
            # ),
        ),
    )
    sampler.sync(
        [
            Trial(
                last_known_state=Trial.State.COMPLETED,
                values=[ObjectiveValue(value=random.random(), status=ObjectiveValue.Status.VALID)],
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(int_value=random.randint(2, 12)),
                        distribution=int_distribution(low=2, high=12),
                    ),
                    "bar": Parameter(
                        value=ParameterValue(double_value=random.random() * 2.0),
                        distribution=float_distribution(low=-2.3, high=3.4),
                    ),
                },
            )
            for _ in range(40)
        ]
    )
    _, _ = sampler.joint_sample(fixed_parameters={})

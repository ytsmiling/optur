import random

import numpy as np
import pytest  # type: ignore

from optur.proto.search_space_pb2 import Distribution, ParameterValue
from optur.proto.study_pb2 import Parameter, Trial
from optur.samplers.tpe import _MixturedDistribution


def int_distribution(low: int, high: int, log_scale: bool = False) -> Distribution:
    return Distribution(
        int_distribution=Distribution.IntDistribution(low=low, high=high, log_scale=log_scale)
    )


@pytest.mark.parametrize("log_scale", [True, False])
def test_int_distribution_samples_valid_values(log_scale: bool) -> None:
    dist = _MixturedDistribution(
        name="foo",
        distribution=int_distribution(low=1, high=100, log_scale=log_scale),
        trials=[
            Trial(
                parameters={
                    "foo": Parameter(value=ParameterValue(int_value=random.randint(10, 30)))
                }
            )
            for _ in range(100)
        ],
        n_distribution=1,
        weights=np.ones(100) * 0.01,
    )
    active_indices = np.asarray(range(1, 100, 2))
    samples = dist.sample(active_indices=active_indices)
    assert samples.dtype == np.dtype("int64")
    assert len(samples) == len(active_indices)
    assert (1 <= samples).all()
    assert (samples <= 100).all()

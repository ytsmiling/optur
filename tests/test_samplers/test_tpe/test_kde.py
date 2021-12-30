import random

import numpy as np
import pytest

from optur.proto.search_space_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import Parameter, Trial
from optur.samplers.tpe import _UnivariateKDE


def int_distribution(low: int, high: int, log_scale: bool = False) -> Distribution:
    return Distribution(
        int_distribution=Distribution.IntDistribution(low=low, high=high, log_scale=log_scale)
    )


def float_distribution(low: float, high: float, log_scale: bool = False) -> Distribution:
    return Distribution(
        float_distribution=Distribution.FloatDistribution(low=low, high=high, log_scale=log_scale)
    )


@pytest.mark.parametrize("int_log_scale", [True, False])
@pytest.mark.parametrize("float_log_scale", [True, False])
def test_univariate_kde_samples_valid_samples(int_log_scale: bool, float_log_scale: bool) -> None:
    weights = np.random.random(99)
    kde = _UnivariateKDE(
        search_space=SearchSpace(
            distributions={
                "foo": float_distribution(low=10.0, high=220.0, log_scale=int_log_scale),
                "bar": int_distribution(low=1, high=9, log_scale=float_log_scale),
            }
        ),
        trials=[
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(double_value=random.random() * 100.0 + 50.0)
                    ),
                    "bar": Parameter(value=ParameterValue(int_value=random.randint(3, 5))),
                }
            )
            for _ in range(100)
        ],
        weights=weights / weights.sum(),
    )
    samples = kde.sample(fixed_parameters={}, k=17)
    assert set(samples.keys()) == {"foo", "bar"}
    assert samples["foo"].shape == (17,)
    assert samples["foo"].dtype == np.dtype("float64")
    assert (10.0 <= samples["foo"]).all()  # type: ignore
    assert (samples["foo"] <= 220.0).all()  # type: ignore
    assert samples["bar"].shape == (17,)
    assert samples["bar"].dtype == np.dtype("int64")
    assert (1 <= samples["bar"]).all()  # type: ignore
    assert (samples["bar"] <= 9).all()  # type: ignore


@pytest.mark.parametrize("int_log_scale", [True, False])
@pytest.mark.parametrize("float_log_scale", [True, False])
def test_univariate_kde_calcuates_valid_pdf(int_log_scale: bool, float_log_scale: bool) -> None:
    weights = np.random.random(99)
    kde = _UnivariateKDE(
        search_space=SearchSpace(
            distributions={
                "foo": float_distribution(low=10.0, high=220.0, log_scale=int_log_scale),
                "bar": int_distribution(low=1, high=9, log_scale=float_log_scale),
            }
        ),
        trials=[
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(double_value=random.random() * 100.0 + 50.0)
                    ),
                    "bar": Parameter(value=ParameterValue(int_value=random.randint(3, 5))),
                }
            )
            for _ in range(99)
        ],
        weights=weights / weights.sum(),
    )
    samples = kde.sample(fixed_parameters={}, k=17)
    log_pdf = kde.log_pdf(samples)
    assert log_pdf.shape == (17,)
    assert log_pdf.dtype == np.dtype("float64")  # type: ignore
    assert (np.exp(log_pdf) <= 1.0).all()

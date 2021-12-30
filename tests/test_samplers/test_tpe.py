import random

import numpy as np
import pytest

from optur.proto.search_space_pb2 import Distribution, ParameterValue
from optur.proto.study_pb2 import Parameter, Trial
from optur.samplers.tpe import _MixturedDistribution


def int_distribution(low: int, high: int, log_scale: bool = False) -> Distribution:
    return Distribution(
        int_distribution=Distribution.IntDistribution(low=low, high=high, log_scale=log_scale)
    )


def float_distribution(low: int, high: int, log_scale: bool = False) -> Distribution:
    return Distribution(
        float_distribution=Distribution.FloatDistribution(low=low, high=high, log_scale=log_scale)
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
            for _ in range(97)
        ],
        n_distribution=1,
    )
    active_indices = np.asarray(range(1, 97, 2))
    samples = dist.sample(active_indices=active_indices)
    assert samples.dtype == np.dtype("int64")
    assert len(samples) == len(active_indices)
    assert (1 <= samples).all()  # type: ignore
    assert (samples <= 100).all()  # type: ignore


@pytest.mark.parametrize("log_scale", [True, False])
def test_int_distribution_calculates_valid_log_pdf(log_scale: bool) -> None:
    dist = _MixturedDistribution(
        name="foo",
        distribution=int_distribution(low=1, high=100, log_scale=log_scale),
        trials=[
            Trial(
                parameters={
                    "foo": Parameter(value=ParameterValue(int_value=random.randint(10, 30)))
                }
            )
            for _ in range(97)
        ],
        n_distribution=1,
    )
    active_indices = np.asarray(range(1, 97, 2))
    samples = dist.sample(active_indices=active_indices)
    log_pdf = dist.log_pdf(samples)
    assert log_pdf.dtype == np.dtype("float64")  # type: ignore
    assert log_pdf.shape == (len(samples), 97)
    assert (np.exp(log_pdf) <= 1.0).all()
    assert (
        np.exp(dist.log_pdf(np.random.randint(10, 30, size=100))).mean()
        > np.exp(dist.log_pdf(np.random.randint(50, 80, size=100))).mean()
    )


@pytest.mark.parametrize("log_scale", [True, False])
def test_float_distribution_samples_valid_values(log_scale: bool) -> None:
    dist = _MixturedDistribution(
        name="foo",
        distribution=float_distribution(low=1, high=100, log_scale=log_scale),
        trials=[
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(double_value=random.random() * 20.0 + 10.0)
                    )
                }
            )
            for _ in range(97)
        ],
        n_distribution=1,
    )
    active_indices = np.asarray(range(1, 97, 2))
    samples = dist.sample(active_indices=active_indices)
    assert samples.dtype == np.dtype("float64")
    assert len(samples) == len(active_indices)
    assert (1.0 <= samples).all()  # type: ignore
    assert (samples <= 100.0).all()  # type: ignore


@pytest.mark.parametrize("log_scale", [True, False])
def test_float_distribution_calculates_valid_log_pdf(log_scale: bool) -> None:
    dist = _MixturedDistribution(
        name="foo",
        distribution=float_distribution(low=1, high=100, log_scale=log_scale),
        trials=[
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(double_value=random.random() * 20.0 + 10.0)
                    )
                }
            )
            for _ in range(97)
        ],
        n_distribution=1,
    )
    active_indices = np.asarray(range(1, 97, 2))
    samples = dist.sample(active_indices=active_indices)
    log_pdf = dist.log_pdf(samples)
    assert log_pdf.dtype == np.dtype("float64")  # type: ignore
    assert log_pdf.shape == (len(samples), 97)
    assert (np.exp(log_pdf) <= 1.0).all()
    assert (
        np.exp(dist.log_pdf(np.asarray([random.random() * 20.0 + 10.0]))).mean()
        > np.exp(dist.log_pdf(np.asarray([random.random() * 20.0 + 60.0]))).mean()
    )

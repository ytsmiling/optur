from typing import Sequence

from optur.proto.search_space_pb2 import Distribution, ParameterValue
from optur.search_space.distribution import (
    are_identical_distributions,
    does_distribution_contain_value,
)


def int_distribution(low: int, high: int, log_scale: bool = False) -> Distribution:
    return Distribution(
        int_distribution=Distribution.IntDistribution(low=low, high=high, log_scale=log_scale)
    )


def float_distribution(low: float, high: float, log_scale: bool = False) -> Distribution:
    return Distribution(
        float_distribution=Distribution.FloatDistribution(low=low, high=high, log_scale=log_scale)
    )


def categorical_distribution(choices: Sequence[ParameterValue]) -> Distribution:
    return Distribution(
        categorical_distribution=Distribution.CategoricalDistribution(choices=choices)
    )


def fixed_distribution(values: Sequence[ParameterValue]) -> Distribution:
    return Distribution(fixed_distribution=Distribution.FixedDistribution(values=values))


def test_int_distribution_comparison() -> None:
    assert are_identical_distributions(
        a=int_distribution(low=1, high=2, log_scale=False),
        b=int_distribution(low=1, high=2, log_scale=False),
    )
    assert not are_identical_distributions(
        a=int_distribution(low=1, high=2, log_scale=False),
        b=int_distribution(low=1, high=3, log_scale=False),
    )
    assert not are_identical_distributions(
        a=int_distribution(low=1, high=2, log_scale=False),
        b=int_distribution(low=1, high=2, log_scale=True),
    )
    assert not are_identical_distributions(
        a=int_distribution(low=1, high=2, log_scale=False),
        b=float_distribution(low=1.0, high=2.0, log_scale=False),
    )


def test_float_distribution_comparison() -> None:
    assert are_identical_distributions(
        a=float_distribution(low=1.1, high=2.2, log_scale=False),
        b=float_distribution(low=1.1, high=2.2, log_scale=False),
    )
    assert not are_identical_distributions(
        a=float_distribution(low=1.1, high=3.0, log_scale=False),
        b=float_distribution(low=1.1, high=3.1, log_scale=False),
    )
    assert not are_identical_distributions(
        a=float_distribution(low=1.1, high=2.0, log_scale=False),
        b=float_distribution(low=1.1, high=2.0, log_scale=True),
    )
    assert not are_identical_distributions(
        a=int_distribution(low=1, high=2, log_scale=False),
        b=float_distribution(low=1.0, high=2.0, log_scale=False),
    )


def test_categorical_distribution_comparison() -> None:
    assert are_identical_distributions(
        a=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
    )
    assert are_identical_distributions(
        a=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=categorical_distribution(
            choices=[
                ParameterValue(double_value=2.0),
                ParameterValue(int_value=1),
                ParameterValue(string_value="foo"),
            ]
        ),
    )
    assert not are_identical_distributions(
        a=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=categorical_distribution(
            choices=[
                ParameterValue(int_value=2),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
    )
    assert not are_identical_distributions(
        a=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=1.0),
                ParameterValue(string_value="foo"),
            ]
        ),
    )
    assert not are_identical_distributions(
        a=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="bar"),
            ]
        ),
    )


def test_fixed_distribution_comparison() -> None:
    assert are_identical_distributions(
        a=fixed_distribution(values=[ParameterValue(int_value=1)]),
        b=fixed_distribution(values=[ParameterValue(int_value=1)]),
    )
    assert not are_identical_distributions(
        a=fixed_distribution(values=[ParameterValue(int_value=1)]),
        b=fixed_distribution(values=[ParameterValue(int_value=2)]),
    )
    assert are_identical_distributions(
        a=fixed_distribution(values=[ParameterValue(double_value=1.0)]),
        b=fixed_distribution(values=[ParameterValue(double_value=1.0)]),
    )
    assert not are_identical_distributions(
        a=fixed_distribution(values=[ParameterValue(double_value=1.0)]),
        b=fixed_distribution(values=[ParameterValue(double_value=2.0)]),
    )
    assert are_identical_distributions(
        a=fixed_distribution(values=[ParameterValue(string_value="foo")]),
        b=fixed_distribution(values=[ParameterValue(string_value="foo")]),
    )
    assert not are_identical_distributions(
        a=fixed_distribution(values=[ParameterValue(string_value="foo")]),
        b=fixed_distribution(values=[ParameterValue(string_value="bar")]),
    )
    assert are_identical_distributions(
        a=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
    )
    assert are_identical_distributions(
        a=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=fixed_distribution(
            values=[
                ParameterValue(double_value=2.0),
                ParameterValue(int_value=1),
                ParameterValue(string_value="foo"),
            ]
        ),
    )


def test_int_distribution_contains_check() -> None:
    assert does_distribution_contain_value(
        distribution=int_distribution(low=1, high=3),
        value=ParameterValue(int_value=1),
    )
    assert does_distribution_contain_value(
        distribution=int_distribution(low=1, high=3),
        value=ParameterValue(int_value=2),
    )
    assert does_distribution_contain_value(
        distribution=int_distribution(low=1, high=3),
        value=ParameterValue(int_value=3),
    )
    assert not does_distribution_contain_value(
        distribution=int_distribution(low=1, high=3),
        value=ParameterValue(int_value=0),
    )
    assert not does_distribution_contain_value(
        distribution=int_distribution(low=1, high=3),
        value=ParameterValue(int_value=4),
    )
    assert not does_distribution_contain_value(
        distribution=int_distribution(low=1, high=3),
        value=ParameterValue(double_value=2.0),
    )
    assert not does_distribution_contain_value(
        distribution=int_distribution(low=1, high=3),
        value=ParameterValue(string_value="foo"),
    )


def test_float_distribution_contains_check() -> None:
    assert does_distribution_contain_value(
        distribution=float_distribution(low=1.0, high=3.0),
        value=ParameterValue(double_value=1.0),
    )
    assert does_distribution_contain_value(
        distribution=float_distribution(low=1.0, high=3.0),
        value=ParameterValue(double_value=2.0),
    )
    assert does_distribution_contain_value(
        distribution=float_distribution(low=1.0, high=3.0),
        value=ParameterValue(double_value=3.0),
    )
    assert not does_distribution_contain_value(
        distribution=float_distribution(low=1.0, high=3.0),
        value=ParameterValue(double_value=0.0),
    )
    assert not does_distribution_contain_value(
        distribution=float_distribution(low=1.0, high=3.0),
        value=ParameterValue(double_value=4.0),
    )
    assert not does_distribution_contain_value(
        distribution=float_distribution(low=1.0, high=3.0),
        value=ParameterValue(int_value=2),
    )
    assert not does_distribution_contain_value(
        distribution=float_distribution(low=1.0, high=3.0),
        value=ParameterValue(string_value="foo"),
    )


def test_categorical_distribution_contains_check() -> None:
    assert does_distribution_contain_value(
        distribution=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(int_value=1),
    )
    assert does_distribution_contain_value(
        distribution=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(double_value=2.0),
    )
    assert does_distribution_contain_value(
        distribution=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(string_value="foo"),
    )
    assert not does_distribution_contain_value(
        distribution=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(int_value=2),
    )
    assert not does_distribution_contain_value(
        distribution=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(double_value=1.0),
    )
    assert not does_distribution_contain_value(
        distribution=categorical_distribution(
            choices=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(string_value="bar"),
    )


def test_fixed_distribution_contains_check() -> None:
    assert does_distribution_contain_value(
        distribution=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(int_value=1),
    )
    assert does_distribution_contain_value(
        distribution=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(double_value=2.0),
    )
    assert does_distribution_contain_value(
        distribution=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(string_value="foo"),
    )
    assert not does_distribution_contain_value(
        distribution=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(int_value=2),
    )
    assert not does_distribution_contain_value(
        distribution=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(double_value=1.0),
    )
    assert not does_distribution_contain_value(
        distribution=fixed_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(string_value="bar"),
    )

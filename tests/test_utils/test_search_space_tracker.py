from typing import Sequence

import pytest

from optur.errors import InCompatibleSearchSpaceError
from optur.proto.search_space_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import Parameter, Trial
from optur.utils.search_space_tracker import (
    SearchSpaceTracker,
    are_identical_distributions,
    are_identical_search_spaces,
    does_distribution_contain_value,
    merge_distributions,
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


def fixed_distribution(value: ParameterValue) -> Distribution:
    return Distribution(fixed_distribution=Distribution.FixedDistribution(value=value))


def unknown_distribution(values: Sequence[ParameterValue]) -> Distribution:
    return Distribution(unknown_distribution=Distribution.UnknownDistribution(values=values))


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


def test_unknown_distribution_comparison() -> None:
    assert are_identical_distributions(
        a=unknown_distribution(values=[ParameterValue(int_value=1)]),
        b=unknown_distribution(values=[ParameterValue(int_value=1)]),
    )
    assert not are_identical_distributions(
        a=unknown_distribution(values=[ParameterValue(int_value=1)]),
        b=unknown_distribution(values=[ParameterValue(int_value=2)]),
    )
    assert are_identical_distributions(
        a=unknown_distribution(values=[ParameterValue(double_value=1.0)]),
        b=unknown_distribution(values=[ParameterValue(double_value=1.0)]),
    )
    assert not are_identical_distributions(
        a=unknown_distribution(values=[ParameterValue(double_value=1.0)]),
        b=unknown_distribution(values=[ParameterValue(double_value=2.0)]),
    )
    assert are_identical_distributions(
        a=unknown_distribution(values=[ParameterValue(string_value="foo")]),
        b=unknown_distribution(values=[ParameterValue(string_value="foo")]),
    )
    assert not are_identical_distributions(
        a=unknown_distribution(values=[ParameterValue(string_value="foo")]),
        b=unknown_distribution(values=[ParameterValue(string_value="bar")]),
    )
    assert are_identical_distributions(
        a=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
    )
    assert are_identical_distributions(
        a=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        b=unknown_distribution(
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


def test_unknown_distribution_contains_check() -> None:
    # UnknownDistribution contains Any.
    assert does_distribution_contain_value(
        distribution=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(int_value=1),
    )
    assert does_distribution_contain_value(
        distribution=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(double_value=2.0),
    )
    assert does_distribution_contain_value(
        distribution=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(string_value="foo"),
    )
    assert does_distribution_contain_value(
        distribution=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(int_value=2),
    )
    assert does_distribution_contain_value(
        distribution=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(double_value=1.0),
    )
    assert does_distribution_contain_value(
        distribution=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(double_value=2.0),
                ParameterValue(string_value="foo"),
            ]
        ),
        value=ParameterValue(string_value="bar"),
    )


def test_merge_int_distributions() -> None:
    assert are_identical_distributions(
        merge_distributions(a=int_distribution(low=1, high=3), b=int_distribution(low=1, high=3)),
        int_distribution(low=1, high=3),
    )
    assert are_identical_distributions(
        merge_distributions(
            a=int_distribution(low=1, high=3),
            b=unknown_distribution(values=[ParameterValue(int_value=1)]),
        ),
        int_distribution(low=1, high=3),
    )
    assert are_identical_distributions(
        merge_distributions(
            a=int_distribution(low=1, high=3),
            b=unknown_distribution(
                values=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
        ),
        int_distribution(low=1, high=3),
    )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=int_distribution(low=1, high=3),
            b=unknown_distribution(values=[ParameterValue(int_value=0)]),
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=int_distribution(low=1, high=3),
            b=unknown_distribution(values=[ParameterValue(double_value=2.0)]),
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=int_distribution(low=1, high=3),
            b=unknown_distribution(
                values=[ParameterValue(int_value=2), ParameterValue(int_value=4)]
            ),
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(a=int_distribution(low=1, high=4), b=int_distribution(low=2, high=3))
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(a=int_distribution(low=1, high=4), b=int_distribution(low=0, high=5))
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=int_distribution(low=1, high=3), b=float_distribution(low=1.0, high=3.0)
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=int_distribution(low=1, high=3),
            b=categorical_distribution(choices=[ParameterValue(int_value=1)]),
        )


def test_merge_float_distributions() -> None:
    assert are_identical_distributions(
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0), b=float_distribution(low=1.0, high=3.0)
        ),
        float_distribution(low=1.0, high=3.0),
    )
    assert are_identical_distributions(
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0),
            b=unknown_distribution(values=[ParameterValue(double_value=1.0)]),
        ),
        float_distribution(low=1.0, high=3.0),
    )
    assert are_identical_distributions(
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0),
            b=unknown_distribution(
                values=[ParameterValue(double_value=1.0), ParameterValue(double_value=2.0)]
            ),
        ),
        float_distribution(low=1.0, high=3.0),
    )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0),
            b=unknown_distribution(values=[ParameterValue(double_value=0.0)]),
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0),
            b=unknown_distribution(values=[ParameterValue(int_value=2)]),
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0),
            b=unknown_distribution(
                values=[ParameterValue(double_value=2.0), ParameterValue(double_value=4.0)]
            ),
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=float_distribution(low=1.0, high=4.0), b=float_distribution(low=2.0, high=3.0)
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=float_distribution(low=1.0, high=4.0), b=float_distribution(low=0.0, high=5.0)
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0), b=int_distribution(low=1, high=3)
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=float_distribution(low=1.0, high=3.0),
            b=categorical_distribution(choices=[ParameterValue(double_value=1.0)]),
        )


def test_merge_categorical_distributions() -> None:
    assert are_identical_distributions(
        a=merge_distributions(
            a=categorical_distribution(
                choices=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
            b=categorical_distribution(
                choices=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
        ),
        b=categorical_distribution(
            choices=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
        ),
    )
    assert are_identical_distributions(
        a=merge_distributions(
            a=categorical_distribution(
                choices=[ParameterValue(int_value=1), ParameterValue(double_value=2.0)]
            ),
            b=categorical_distribution(
                choices=[ParameterValue(double_value=2.0), ParameterValue(int_value=1)]
            ),
        ),
        b=categorical_distribution(
            choices=[ParameterValue(int_value=1), ParameterValue(double_value=2.0)]
        ),
    )
    assert are_identical_distributions(
        a=merge_distributions(
            a=categorical_distribution(
                choices=[ParameterValue(int_value=1), ParameterValue(double_value=2.0)]
            ),
            b=unknown_distribution(values=[ParameterValue(int_value=1)]),
        ),
        b=categorical_distribution(
            choices=[ParameterValue(int_value=1), ParameterValue(double_value=2.0)]
        ),
    )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=categorical_distribution(
                choices=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
            b=categorical_distribution(
                choices=[ParameterValue(int_value=1), ParameterValue(int_value=3)]
            ),
        )
    with pytest.raises(InCompatibleSearchSpaceError):
        merge_distributions(
            a=categorical_distribution(
                choices=[ParameterValue(int_value=1), ParameterValue(double_value=2.0)]
            ),
            b=unknown_distribution(values=[ParameterValue(int_value=2)]),
        )


def test_merge_unknown_distributions() -> None:
    assert are_identical_distributions(
        a=merge_distributions(
            a=unknown_distribution(
                values=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
            b=unknown_distribution(
                values=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
        ),
        b=unknown_distribution(values=[ParameterValue(int_value=1), ParameterValue(int_value=2)]),
    )
    assert are_identical_distributions(
        a=merge_distributions(
            a=unknown_distribution(
                values=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
            b=unknown_distribution(
                values=[ParameterValue(int_value=1), ParameterValue(int_value=3)]
            ),
        ),
        b=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(int_value=2),
                ParameterValue(int_value=3),
            ]
        ),
    )
    assert are_identical_distributions(
        a=merge_distributions(
            a=unknown_distribution(
                values=[ParameterValue(int_value=1), ParameterValue(int_value=2)]
            ),
            b=unknown_distribution(
                values=[ParameterValue(double_value=1.0), ParameterValue(string_value="foo")]
            ),
        ),
        b=unknown_distribution(
            values=[
                ParameterValue(int_value=1),
                ParameterValue(int_value=2),
                ParameterValue(double_value=1.0),
                ParameterValue(string_value="foo"),
            ]
        ),
    )


def test_merge_fixed_distributions() -> None:
    # TODO(tsuzuku): Implement this.
    pass


def test_search_space_tracker_merges_unknown_distributions() -> None:
    search_space_tracker = SearchSpaceTracker(
        search_space=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[ParameterValue(int_value=2), ParameterValue(string_value="bar")],
                )
            }
        )
    )
    assert are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[ParameterValue(int_value=2), ParameterValue(string_value="bar")],
                )
            }
        ),
    )
    search_space_tracker.sync(
        [Trial(parameters={"foo": Parameter(value=ParameterValue(double_value=0.2))})]
    )
    assert are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[
                        ParameterValue(int_value=2),
                        ParameterValue(string_value="bar"),
                        ParameterValue(double_value=0.2),
                    ],
                )
            }
        ),
    )
    search_space_tracker.sync(
        [
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(double_value=0.3),
                        distribution=unknown_distribution(
                            values=[ParameterValue(double_value=0.3)]
                        ),
                    )
                }
            )
        ]
    )
    assert are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[
                        ParameterValue(int_value=2),
                        ParameterValue(string_value="bar"),
                        ParameterValue(double_value=0.2),
                        ParameterValue(double_value=0.3),
                    ],
                )
            }
        ),
    )
    search_space_tracker.sync(
        [
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(double_value=0.3),
                        distribution=categorical_distribution(
                            choices=[
                                ParameterValue(int_value=2),
                                ParameterValue(string_value="bar"),
                                ParameterValue(double_value=0.2),
                                ParameterValue(double_value=0.3),
                                ParameterValue(double_value=0.4),
                            ]
                        ),
                    )
                }
            )
        ]
    )
    assert are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={
                "foo": categorical_distribution(
                    choices=[
                        ParameterValue(int_value=2),
                        ParameterValue(string_value="bar"),
                        ParameterValue(double_value=0.2),
                        ParameterValue(double_value=0.3),
                        ParameterValue(double_value=0.4),
                    ],
                )
            }
        ),
    )


def test_search_space_tracker_updates_with_new_parameters() -> None:
    search_space_tracker = SearchSpaceTracker(
        search_space=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[ParameterValue(int_value=2), ParameterValue(string_value="bar")],
                )
            }
        )
    )
    assert are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[ParameterValue(int_value=2), ParameterValue(string_value="bar")],
                )
            }
        ),
    )
    search_space_tracker.sync(
        [Trial(parameters={"bar": Parameter(value=ParameterValue(double_value=0.2))})]
    )
    are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[
                        ParameterValue(int_value=2),
                        ParameterValue(string_value="bar"),
                    ],
                ),
                "bar": unknown_distribution(
                    values=[
                        ParameterValue(double_value=0.2),
                    ],
                ),
            }
        ),
    )


def test_search_space_tracker_promotes_unknown_distribution_to_int_distribution() -> None:
    search_space_tracker = SearchSpaceTracker(
        search_space=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[ParameterValue(int_value=1), ParameterValue(int_value=3)],
                )
            }
        )
    )
    search_space_tracker.sync(
        [
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(int_value=4),
                        distribution=int_distribution(low=1, high=5),
                    )
                }
            )
        ]
    )
    assert are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={"foo": int_distribution(low=1, high=5)},
        ),
    )


def test_search_space_tracker_promotes_unknown_distribution_to_float_distribution() -> None:
    search_space_tracker = SearchSpaceTracker(
        search_space=SearchSpace(
            distributions={
                "foo": unknown_distribution(
                    values=[
                        ParameterValue(double_value=0.1),
                        ParameterValue(double_value=1.3),
                    ],
                )
            }
        )
    )
    search_space_tracker.sync(
        [
            Trial(
                parameters={
                    "foo": Parameter(
                        value=ParameterValue(double_value=1.1),
                        distribution=float_distribution(low=0.01, high=2.5),
                    )
                }
            )
        ]
    )
    assert are_identical_search_spaces(
        a=search_space_tracker.current_search_space,
        b=SearchSpace(
            distributions={"foo": float_distribution(low=0.01, high=2.5)},
        ),
    )


def test_search_space_tracker_raises_on_distribution_conflicts() -> None:
    search_space_tracker = SearchSpaceTracker(
        search_space=SearchSpace(
            distributions={
                "foo": float_distribution(low=0.01, high=2.5),
            }
        )
    )
    with pytest.raises(InCompatibleSearchSpaceError):
        search_space_tracker.sync(
            [Trial(parameters={"foo": Parameter(value=ParameterValue(double_value=3.1))})]
        )
    search_space_tracker = SearchSpaceTracker(
        search_space=SearchSpace(
            distributions={
                "foo": float_distribution(low=0.01, high=2.5),
            }
        )
    )
    with pytest.raises(InCompatibleSearchSpaceError):
        search_space_tracker.sync(
            [Trial(parameters={"foo": Parameter(value=ParameterValue(int_value=2))})]
        )
    search_space_tracker = SearchSpaceTracker(
        search_space=SearchSpace(
            distributions={
                "foo": float_distribution(low=0.01, high=2.5),
            }
        )
    )
    with pytest.raises(InCompatibleSearchSpaceError):
        search_space_tracker.sync(
            [
                Trial(
                    parameters={
                        "foo": Parameter(
                            value=ParameterValue(double_value=2.1),
                            distribution=float_distribution(low=0.0, high=5.0),
                        )
                    }
                )
            ]
        )

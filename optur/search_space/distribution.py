from optur.errors import InCompatibleSearchSpaceError
from optur.proto.search_space_pb2 import Distribution, ParameterValue


def does_distribution_contain_value(distribution: Distribution, value: ParameterValue) -> bool:
    """Check whether the value can be drawn from the distribution."""
    if distribution.HasField("int_distribution"):
        if not value.HasField("int_value"):
            return False
        return (
            distribution.int_distribution.low
            <= value.int_value
            <= distribution.int_distribution.high
        )
    if distribution.HasField("float_distribution"):
        if not value.HasField("double_value"):
            return False
        return (
            distribution.float_distribution.low
            <= value.double_value
            <= distribution.float_distribution.high
        )
    if distribution.HasField("categorical_distribution"):
        return any(value == v for v in distribution.categorical_distribution.choices)
    raise NotImplementedError("")


def are_identical_distributions(a: Distribution, b: Distribution) -> bool:
    """Check whether the two distributions are identical."""
    if a == b:
        # In most usecase, a == b.
        return True
    if a.HasField("categorical_distribution") and b.HasField("categorical_distribution"):
        # When values have different order, categorical distributions do not satisfy ``a == b``
        # even if they are identical.
        values1 = a.categorical_distribution.choices
        values2 = b.categorical_distribution.choices
        if len(values1) == len(values2) and all(any(v1 == v2 for v2 in values2) for v1 in values1):
            return True
    if a.HasField("fixed_distribution") and b.HasField("fixed_distribution"):
        # The same with categorical distribution.
        values1 = a.fixed_distribution.values
        values2 = b.fixed_distribution.values
        if len(values1) == len(values2) and all(any(v1 == v2 for v2 in values2) for v1 in values1):
            return True
    return False


def merge_distributions(
    original_distribution: Distribution,
    new_distribution: Distribution,
) -> Distribution:
    """Merge two distributions into one distribution.

    When one or more distributions are `FixedDistribution`,
    this method merges the two distributions.
    Otherwise, check whether the two distributions are identical.
    When the two distributions are incompatible, `InCompatibleSearchSpaceError` will be raised.
    """
    if original_distribution.HasField("fixed_distribution"):
        if new_distribution.HasField("fixed_distribution"):
            values = [v for v in original_distribution.fixed_distribution.values]
            for new_v in new_distribution.fixed_distribution.values:
                if not any(new_v == v for v in values):
                    values.append(new_v)
            return Distribution(fixed_distribution=Distribution.FixedDistribution(values=values))
        if all(
            does_distribution_contain_value(new_distribution, value)
            for value in original_distribution.fixed_distribution.values
        ):
            return new_distribution
        raise InCompatibleSearchSpaceError("")
    if new_distribution.HasField("fixed_distribution"):
        if all(
            does_distribution_contain_value(original_distribution, value)
            for value in new_distribution.fixed_distribution.values
        ):
            return original_distribution
    if are_identical_distributions(original_distribution, new_distribution):
        return original_distribution
    raise InCompatibleSearchSpaceError("")  # TODO(tsuzuku)

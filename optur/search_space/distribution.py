from optur.errors import InCompatibleSearchSpaceError
from optur.proto.search_space_pb2 import Distribution, ParameterValue


def does_distribution_contain_value(distribution: Distribution, value: ParameterValue) -> bool:
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


def extend_distribution(
    original_distribution: Distribution,
    new_distribution: Distribution,
) -> Distribution:
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
    if original_distribution == new_distribution:
        return original_distribution
    if original_distribution.HasField("categorical_distribution") and new_distribution.HasField(
        "categorical_distribution"
    ):
        values1 = original_distribution.categorical_distribution.choices
        values2 = new_distribution.categorical_distribution.choices
        if len(values1) == len(values2) and all(any(v1 == v2 for v2 in values2) for v1 in values1):
            return original_distribution
    raise InCompatibleSearchSpaceError("")  # TODO(tsuzuku)

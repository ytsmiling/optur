from typing import Optional, Sequence

from google.protobuf.timestamp_pb2 import Timestamp

from optur.errors import InCompatibleSearchSpaceError
from optur.proto.search_space_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import Trial


class SearchSpaceTracker:
    """Infer search space from the past trials.

    This class infers the search space from the past trials.
    This process is called `intersection_searchspace` in optuna, but
    optur does not necessarily infer the same search space with optuna.
    """

    def __init__(self, search_space: Optional[SearchSpace]) -> None:
        self._initial_search_space = search_space
        self._current_search_space = SearchSpace()
        if search_space is not None:
            self._current_search_space.CopyFrom(search_space)
        self._last_update_time: Optional[Timestamp] = None

    @property
    def last_update_time(self) -> Optional[Timestamp]:
        return self._last_update_time

    def update_timestamp(self, timestamp: Timestamp) -> None:
        self._last_update_time = timestamp

    @property
    def current_search_space(self) -> SearchSpace:
        return self._current_search_space

    def contains(self, name: str, value: ParameterValue) -> bool:
        return (
            name not in self._current_search_space.distributions
            or does_distribution_contain_value(
                self._current_search_space.distributions[name], value
            )
        )

    def sync(self, trials: Sequence[Trial]) -> None:
        for trial in trials:
            for name, param in trial.parameters.items():
                if param.HasField("distribution"):
                    pdist = param.distribution
                else:
                    pdist = Distribution(
                        fixed_distribution=Distribution.FixedDistribution(values=[param.value])
                    )
                if name in self._current_search_space.distributions:
                    cur_dist = self._current_search_space.distributions[name]
                    if not are_identical_distributions(pdist, cur_dist):
                        try:
                            new_search_space = merge_distributions(pdist, cur_dist)
                            self._current_search_space.distributions[name].CopyFrom(
                                new_search_space
                            )
                        except InCompatibleSearchSpaceError as e:
                            raise InCompatibleSearchSpaceError("") from e  # TODO(tsuzuku)
                else:
                    self._current_search_space.distributions[name].CopyFrom(pdist)


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
    if distribution.HasField("fixed_distribution"):
        return any(value == v for v in distribution.fixed_distribution.values)
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
    a: Distribution,
    b: Distribution,
) -> Distribution:
    """Merge two distributions into one distribution.

    When one or more distributions are `FixedDistribution`,
    this method merges the two distributions.
    Otherwise, check whether the two distributions are identical.
    When the two distributions are incompatible, `InCompatibleSearchSpaceError` will be raised.
    """
    if a.HasField("fixed_distribution"):
        if b.HasField("fixed_distribution"):
            values = [v for v in a.fixed_distribution.values]
            for new_v in b.fixed_distribution.values:
                if not any(new_v == v for v in values):
                    values.append(new_v)
            return Distribution(fixed_distribution=Distribution.FixedDistribution(values=values))
        if all(does_distribution_contain_value(b, value) for value in a.fixed_distribution.values):
            return b
        raise InCompatibleSearchSpaceError("")
    if b.HasField("fixed_distribution"):
        if all(does_distribution_contain_value(a, value) for value in b.fixed_distribution.values):
            return a
    if are_identical_distributions(a, b):
        return a
    raise InCompatibleSearchSpaceError("")  # TODO(tsuzuku)

from typing import Dict, Sequence, Union, overload

from optur.proto.study_pb2 import Distribution, ParameterValue
from optur.proto.study_pb2 import Trial as TrialProto
from optur.samplers import Sampler
from optur.storages import StorageClient


class Trial:
    def __init__(self, trial_proto: TrialProto, storage: StorageClient, sampler: Sampler) -> None:
        self._initial_trial_proto = trial_proto
        self._trial_proto = TrialProto()
        self._trial_proto.CopyFrom(trial_proto)
        self._storage = storage
        self._sampler = sampler
        self._suggested_parameters: Dict[str, ParameterValue] = {}

    def get_proto(self) -> TrialProto:
        """Convert this trial to :class:`~optur.proto.study_pb2.Trial`."""
        ret = TrialProto()
        ret.CopyFrom(self._trial_proto)
        return ret

    def suggest_int(self, name: str, low: int, high: int, *, log_scale: bool) -> int:
        """Suggest int parameter."""
        if name in self._trial_proto.parameters:
            value = self._trial_proto.parameters[name].value
            if not value.HasField("int_value"):
                raise RuntimeError()  # TODO(tsuzuku)
            return value.int_value
        value = self._sampler.sample(
            distribution=Distribution(
                int_distribution=Distribution.IntDistribution(
                    low=low, high=high, log_scale=log_scale
                )
            )
        )
        assert value.HasField("int_value")
        self._trial_proto.parameters[name].value.CopyFrom(value)
        return value.int_value

    def suggest_float(self, name: str, low: float, high: float, *, log_scale: bool) -> float:
        """Suggest float parameter."""
        if name in self._trial_proto.parameters:
            value = self._trial_proto.parameters[name].value
            if not value.HasField("double_value"):
                raise RuntimeError()  # TODO(tsuzuku)
            return value.double_value
        value = self._sampler.sample(
            distribution=Distribution(
                float_distribution=Distribution.FloatDistribution(
                    low=low, high=high, log_scale=log_scale
                )
            )
        )
        assert value.HasField("double_value")
        self._trial_proto.parameters[name].value.CopyFrom(value)
        return value.double_value

    def suggest_categorical(
        self, name: str, choices: Sequence[Union[int, float, str]]
    ) -> Union[int, float, str]:
        """Suggest categorical parameter."""
        if name in self._trial_proto.parameters:
            value = self._trial_proto.parameters[name].value
        else:
            value = self._sampler.sample(
                distribution=Distribution(
                    categorical_distribution=Distribution.CategoricalDistribution(
                        choices=[]  # TODO(tsuzuku): Implement this.
                    )
                )
            )
        self._trial_proto.parameters[name].value.CopyFrom(value)
        return value.double_value  # TODO(tsuzuku): Handle other types.

    @overload
    def set_parameter(self, name: str, parameter: int, *, force: bool = False) -> int:
        ...

    @overload
    def set_parameter(self, name: str, parameter: float, *, force: bool = False) -> float:
        ...

    @overload
    def set_parameter(self, name: str, parameter: str, *, force: bool = False) -> str:
        ...

    def set_parameter(
        self, name: str, parameter: Union[int, float, str], *, force: bool = False
    ) -> Union[int, float, str]:
        pass

    def clear_parameter(self, name: str, *, force: bool) -> bool:
        pass

    def update_parameters(self, parameters: Dict[str, ParameterValue]) -> None:
        pass

    def clear(self, *, hard: bool, reload: bool) -> None:
        pass

    def flush(self) -> None:
        """Write this trial to the storage."""
        pass

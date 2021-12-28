from typing import Dict, Sequence, Union, overload

from optur.proto.search_space_pb2 import Distribution, ParameterValue
from optur.proto.study_pb2 import Parameter, StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.samplers import Sampler
from optur.storages import StorageClient


class Trial:
    def __init__(
        self,
        trial_proto: TrialProto,
        study_info: StudyInfo,
        storage: StorageClient,
        sampler: Sampler,
    ) -> None:
        self._initial_trial_proto = trial_proto
        self._trial_proto = TrialProto()
        self._trial_proto.CopyFrom(trial_proto)
        self._study_info = study_info
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
                        choices=[_value_to_parameter_value(choice) for choice in choices]
                    )
                )
            )
        self._trial_proto.parameters[name].value.CopyFrom(value)
        return _parameter_value_to_value(value)

    @overload
    def set_parameter(self, name: str, value: int, *, force: bool = False) -> int:
        ...

    @overload
    def set_parameter(self, name: str, value: float, *, force: bool = False) -> float:
        ...

    @overload
    def set_parameter(self, name: str, value: str, *, force: bool = False) -> str:
        ...

    def set_parameter(
        self, name: str, value: Union[int, float, str], *, force: bool = False
    ) -> Union[int, float, str]:
        if not force and name in self._trial_proto.parameters:
            parameter_value = self._trial_proto.parameters[name].value
            if name in self._suggested_parameters:
                del self._suggested_parameters[name]
            return parameter_value.int_value  # TODO(tsuzuku): Handle other types.
        else:
            # TODO(tsuzuku): Set value.
            self._trial_proto.parameters[name].CopyFrom(Parameter(value=ParameterValue()))
            return value

    def clear_parameter(self, name: str, *, force: bool) -> bool:
        if name in self._suggested_parameters:
            del self._suggested_parameters[name]
        if name in self._trial_proto.parameters:
            if name not in self._initial_trial_proto.parameters:
                del self._trial_proto.parameters[name]
                return True
            if force and name in self._initial_trial_proto.parameters:
                del self._trial_proto.parameters[name]
                del self._initial_trial_proto.parameters[name]
                return True
            return False
        return True

    def update_parameters(self, parameters: Dict[str, ParameterValue]) -> None:
        pass

    def clear(self, *, hard: bool, reload: bool) -> None:
        if hard:
            self._initial_trial_proto.parameters.clear()
            del self._initial_trial_proto.values[:]
            self._initial_trial_proto.user_attrs.clear()
            self._initial_trial_proto.system_attrs.clear()
        self._trial_proto.CopyFrom(self._initial_trial_proto)
        if reload:
            timestamp = self._storage.get_current_timestamp()
            trials = self._storage.get_trials(
                study_id=self._study_info.study_id,
                timestamp=self._sampler.last_update_time,
            )
            self._sampler.sync(trials)
            self._sampler.update_timestamp(timestamp)
            self._suggested_parameters = self._sampler.joint_sample(
                fixed_parameters={
                    key: param.value for key, param in self._initial_trial_proto.parameters.items()
                }
            )

    def flush(self) -> None:
        """Write this trial to the storage."""
        self._storage.write_trial(self._trial_proto)


def _value_to_parameter_value(value: Union[int, float, str]) -> ParameterValue:
    pass


def _parameter_value_to_value(value: ParameterValue) -> Union[int, float, str]:
    if value.HasField("int_value"):
        return value.int_value
    if value.HasField("double_value"):
        return value.double_value
    if value.HasField("string_value"):
        return value.string_value
    raise ValueError("")  # TODO(tsuzuku)

from typing import Dict, Sequence, Union, overload

from optur.proto.study_pb2 import ParameterValue
from optur.proto.study_pb2 import Trial as TrialProto
from optur.storages import StorageClient


class Trial:
    def __init__(self, trial_proto: TrialProto, storage: StorageClient) -> None:
        self._trial_proto = trial_proto
        self._storage = storage
        self._suggested_parameters: Dict[str, ParameterValue] = {}
        self._used_parameters = {key: param for key, param in trial_proto.parameters.items()}

    def get_proto(self) -> TrialProto:
        # TODO(tsuzuku): Update the content of the trial proto.
        return self._trial_proto

    def suggest_int(self, name: str, low: int, high: int, *, log_scale: bool) -> int:
        pass

    def suggest_float(self, name: str, low: float, high: float, *, log_scale: bool) -> float:
        pass

    def suggest_categorical(
        self, name: str, choices: Sequence[Union[int, float, str]]
    ) -> Union[int, float, str]:
        pass

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

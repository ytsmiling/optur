from typing import Dict

from optur.proto.study_pb2 import ParameterValue
from optur.proto.study_pb2 import Trial as TrialProto
from optur.storages import StorageClient


class Trial:
    def __init__(self, trial_proto: TrialProto, storage: StorageClient) -> None:
        self._trial_proto = trial_proto
        self._storage = storage

    @property
    def proto(self) -> TrialProto:
        return self._trial_proto

    def update_parameters(self, parameters: Dict[str, ParameterValue]) -> None:
        pass

    def flush(self) -> None:
        """Write this trial to the storage."""
        pass

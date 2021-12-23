from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.study_pb2 import ObjectiveValue
from optur.proto.study_pb2 import Trial as TrialProto
from optur.samplers import Sampler
from optur.storages import Storage
from optur.trial import Trial

ObjectiveFuncType = Callable[[Trial], Union[float, Sequence[float]]]


class _Study:
    def _ask(
        self,
        sampler: Sampler,
        storage: Storage,
        client_id: str,
        last_updated_time: Timestamp,
    ) -> Tuple[Trial, Timestamp]:
        pass

    def optimize(
        self,
        objective: ObjectiveFuncType,
        client_id: str,
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[[Trial], None]]] = None,
    ) -> None:
        pass

    def _run_trials(
        self,
        objective: ObjectiveFuncType,
        client_id: str,
        storage_client: Storage,
        n_trials: Optional[int] = None,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[[Trial], None]]] = None,
    ) -> None:
        # Instantiate sampler.
        # Run _run_trial sequentially.
        pass

    def _run_trial(
        self,
        objective: ObjectiveFuncType,
        sampler: Sampler,
        storage_client: Storage,
        client_id: str,
        last_update_time: Timestamp,
        catch: Tuple[Type[Exception], ...],
        callbacks: Optional[List[Callable[[Trial], None]]] = None,
    ) -> None:
        # Ask.
        # Objective.
        # Tell.
        pass


class Study(_Study):
    def ask(self) -> Trial:
        pass

    def tell(
        self,
        trial_id: str,
        state: int,  # `TrialProto.State` is not a valid type.
        values: Union[Sequence[ObjectiveValue], Dict[str, ObjectiveValue]],
        *,
        study_id: str,  # Make operation faster when there are no index of trial-id.
    ) -> None:
        # Read trial
        # Update trial
        # Write trial
        pass

    def add_trial(self, trial: TrialProto) -> None:
        pass

    def enqueue_trial(self, trial: TrialProto) -> None:
        pass

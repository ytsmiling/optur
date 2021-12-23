import itertools
from collections.abc import Sequence as SequenceType
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from google.protobuf.timestamp_pb2 import Timestamp

from optur.errors import PrunedException
from optur.proto.sampler_pb2 import SamplerConfig
from optur.proto.study_pb2 import ObjectiveValue, StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.proto.study_pb2 import WorkerID
from optur.samplers import Sampler, create_sampler
from optur.storages import Storage, StorageClient
from optur.trial import Trial

ObjectiveFuncType = Callable[[Trial], Union[float, Sequence[float]]]


class _Study:
    def __init__(
        self,
        study_info: StudyInfo,
        sampler_config: SamplerConfig,
        storage: Storage,
    ) -> None:
        self._study_info = study_info
        self._sampler_config = sampler_config
        self._storage = storage

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
        worker_id: WorkerID,
        storage_client: StorageClient,
        n_trials: Optional[int],
        catch: Tuple[Type[Exception], ...],
        callbacks: Optional[List[Callable[[Trial], None]]],
    ) -> None:
        sampler = create_sampler(sampler_config=self._sampler_config)
        trial_counter = itertools.count() if n_trials is None else range(n_trials)
        for _ in trial_counter:
            self._run_trial(
                objective=objective,
                sampler=sampler,
                storage_client=storage_client,
                worker_id=worker_id,
                catch=catch,
                callbacks=callbacks,
            )

    def _run_trial(
        self,
        objective: ObjectiveFuncType,
        sampler: Sampler,
        storage_client: StorageClient,
        worker_id: WorkerID,
        catch: Tuple[Type[Exception], ...],
        callbacks: Optional[List[Callable[[Trial], None]]],
    ) -> None:
        trial = _ask(
            study_id=self._study_info.study_id,
            sampler=sampler,
            storage=storage_client,
            worker_id=worker_id,
        )
        try:
            values = objective(trial)
        except PrunedException:
            trial.proto.last_known_state = TrialProto.State.PRUNED
        except catch:
            trial.proto.last_known_state = TrialProto.State.FAILED
        else:
            # TODO(tsuzuku): Check trial's state in more detail.
            trial.proto.last_known_state = TrialProto.State.COMPLETED
            if isinstance(values, SequenceType):
                del trial.proto.values[:]
                trial.proto.values.extend([ObjectiveValue(value=value) for value in values])
            else:
                del trial.proto.values[:]
                # TODO(tsuzuku): Set values' status.
                trial.proto.values.append(ObjectiveValue(value=float(values)))
        if callbacks:
            for callback in callbacks:
                callback(trial)
        trial.flush()


class Study(_Study):
    def __init__(
        self,
        study_info: StudyInfo,
        storage: Storage,
        sampler: Sampler,
        client_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            study_info=study_info,
            storage=storage,
            sampler_config=sampler.to_sampler_config(),
        )
        self._storage = storage
        # The following three (sampler, client_id, last_update_time) should not be shared by
        # multiple threads or processes.
        # Thus, they are re-instantiated in Study.optimize() function.
        # These three are instantated here for study.ask() and study.tell() APIs.
        self._sampler = sampler
        self._client_id = client_id
        self._last_update_time = Timestamp(seconds=0, nanos=0)

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


def _ask(
    study_id: str,
    sampler: Sampler,
    storage: StorageClient,
    worker_id: WorkerID,
) -> Trial:
    # Sync sampler and storage.
    new_timestamp = storage.get_current_timestamp()
    trials = storage.get_trials(
        study_id=study_id,
        timestamp=sampler.last_update_time,
    )
    sampler.sync(trials)
    sampler.update_timestamp(new_timestamp)
    # Check waiting trials.
    initial_trial: Optional[TrialProto] = None
    for trial in trials:
        if trial.last_known_state == TrialProto.State.WAITING and trial.worker_id == worker_id:
            initial_trial = trial
            break
    if initial_trial is None:
        # TODO(tsuzuku): Create trial object in storage.
        pass
    assert initial_trial is not None
    # Call joint_sample of sampler
    ret = Trial(trial_proto=initial_trial, storage=storage)
    ret.update_parameters(sampler.joint_sample(initial_trial))
    return ret

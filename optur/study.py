import concurrent.futures
import itertools
import math
import uuid
from collections.abc import Sequence as SequenceType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

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


class Study:
    def __init__(
        self,
        study_info: StudyInfo,
        storage: Storage,
        sampler: Sampler,
        client_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._study_info = study_info
        self._storage = storage
        self._client_id = client_id or uuid.uuid4().hex
        # The following two (sampler, last_update_time) should not be shared by
        # multiple threads or processes.
        # Thus, they are re-instantiated in Study.optimize() function.
        # These three are instantated here for study.ask() and study.tell() APIs.
        self._sampler = sampler
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

    def optimize(
        self,
        objective: ObjectiveFuncType,
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[[Trial], None]]] = None,
    ) -> None:
        _optimize(
            objective=objective,
            study_info=self._study_info,
            sampler_config=self._sampler.to_sampler_config(),
            client_id=self._client_id,
            storage=self._storage,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=callbacks,
        )

    def add_trial(self, trial: TrialProto) -> None:
        pass

    def enqueue_trial(self, trial: TrialProto) -> None:
        pass


class _TrialQueue:
    """Trial queue for managing WAITING trials."""

    def __init__(self, states: "Sequence[TrialProto.StateValue]", worker_id: WorkerID) -> None:
        self._trials: Dict[str, TrialProto] = {}
        self._timestamp: Optional[Timestamp] = None
        self._states = states
        self._worker_id = worker_id

    def _is_target_trial(self, trial: TrialProto) -> bool:
        return trial.last_known_state in self._states and _does_own_trial(
            self._worker_id, trial.worker_id
        )

    @property
    def last_update_time(self) -> Optional[Timestamp]:
        return self._timestamp

    def update_timestamp(self, timestamp: Optional[Timestamp]) -> None:
        self._timestamp = timestamp

    def sync(self, trials: Sequence[TrialProto]) -> None:
        for trial in trials:
            if trial.trial_id in self._trials:
                if self._is_target_trial(trial):
                    self._trials[trial.trial_id] = trial
                else:
                    del self._trials[trial.trial_id]
            else:
                if self._is_target_trial(trial):
                    self._trials[trial.trial_id] = trial

    def get_trial(self, state: "TrialProto.StateValue") -> Optional[TrialProto]:
        for trial in self._trials.values():
            if trial.last_known_state == state:
                del self._trials[trial.trial_id]
                return trial
        return None


def _does_own_trial(owner_worker_id: WorkerID, trial_worker_id: WorkerID) -> bool:
    if owner_worker_id.client_id != trial_worker_id.client_id:
        return False
    return owner_worker_id.thread_id == 0 or owner_worker_id == trial_worker_id


def _ask(
    study_info: StudyInfo,
    sampler: Sampler,
    storage: StorageClient,
    trial_queue: _TrialQueue,
    worker_id: WorkerID,
) -> Trial:
    """Ask method.

    This method must do the following.
    * sync the sampler with the storage.
    * sync the waiting trial queue with the storage.
    * write the new or fetched trial to the storage (if required).
    * call joint_sample of the sampler and set to the trial.
    """
    # Sync trial_queue and storage.
    queue_timestamp = trial_queue.last_update_time
    new_timestamp = storage.get_current_timestamp()
    trials = storage.get_trials(study_id=study_info.study_id, timestamp=queue_timestamp)
    trial_queue.sync(trials=trials)
    trial_queue.update_timestamp(timestamp=new_timestamp)
    # Get waiting trial if exists.
    initial_trial = trial_queue.get_trial(state=TrialProto.State.WAITING)
    if initial_trial is None:
        initial_trial = TrialProto(
            trial_id=uuid.uuid4().hex,
            create_time=new_timestamp,
            last_update_time=new_timestamp,
            last_known_state=TrialProto.State.CREATED,
            worker_id=worker_id,
        )
    # Sync sampler and storage.
    sampler_timestamp = sampler.last_update_time
    if queue_timestamp != sampler_timestamp:
        new_timestamp = storage.get_current_timestamp()
        trials = storage.get_trials(study_id=study_info.study_id, timestamp=sampler_timestamp)
    sampler.sync(trials=trials)
    sampler.update_timestamp(timestamp=new_timestamp)
    # TODO(tsuzuku): Persist trial when necessary.
    # Call joint_sample of sampler
    ret = Trial(trial_proto=initial_trial, study_info=study_info, storage=storage, sampler=sampler)
    ret.reset(hard=False, reload=False)
    return ret


def _optimize(
    objective: ObjectiveFuncType,
    study_info: StudyInfo,
    sampler_config: SamplerConfig,
    client_id: str,
    storage: Storage,
    n_trials: Optional[int],
    timeout: Optional[float],
    n_jobs: int,
    catch: Tuple[Type[Exception], ...],
    callbacks: Optional[Sequence[Callable[[Trial], None]]],
) -> None:
    if n_jobs > 1:
        # Storage instance cannot be shared by multiple threads
        # or processes. See :class:`~optur.storage.Storage`'s
        # classdoc for more details.
        clients = [(idx + 1, storage.create_client()) for idx in range(n_jobs)]
    else:
        # Avoid using storage's clients to reduce runtime overhead.
        # TODO(tsuzuku): Benchmark.
        clients = [(0, storage)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures: List[concurrent.futures.Future[Any]] = []
        for thread_id, client in clients:
            # Prefer submit over map for readability.
            future = executor.submit(
                _run_trials,
                objective=objective,
                study_info=study_info,
                sampler_config=sampler_config,
                worker_id=WorkerID(client_id=client_id, thread_id=thread_id),
                storage_client=client,
                n_trials=n_trials,
                catch=catch,
                callbacks=callbacks,
            )
            futures.append(future)
        try:
            for future in futures:
                future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # TODO(tsuzuku): Log a timeout message.
            pass


def _run_trials(
    objective: ObjectiveFuncType,
    study_info: StudyInfo,
    sampler_config: SamplerConfig,
    worker_id: WorkerID,
    storage_client: StorageClient,
    n_trials: Optional[int],
    catch: Tuple[Type[Exception], ...],
    callbacks: Optional[Sequence[Callable[[Trial], None]]],
) -> None:
    print("foo", flush=True)
    # We need to create sampler instances per thread because
    # they are neither thread-safe nor process-safe.
    # Additionally, if we share sampler instances among workers,
    # we have risk that samplers' caches are updated during
    # trials' lifetime.
    # Unlike optuna, it is less likely that the update breaks
    # sampler algorithms in optur, but still, we want to ensure that samplers
    # see the same cache in all `joint_sample` and `sample` calls for the same trial.
    sampler = create_sampler(sampler_config=sampler_config)
    # We also need to create _TrialQueue per thread/process becasue it's associated
    # with worker-id.
    trial_queue = _TrialQueue([TrialProto.State.WAITING], worker_id=worker_id)
    trial_counter = itertools.count() if n_trials is None else range(n_trials)
    for _ in trial_counter:
        _run_trial(
            study_info=study_info,
            objective=objective,
            sampler=sampler,
            storage_client=storage_client,
            worker_id=worker_id,
            catch=catch,
            callbacks=callbacks,
            trial_queue=trial_queue,
        )


def _run_trial(
    objective: ObjectiveFuncType,
    study_info: StudyInfo,
    sampler: Sampler,
    storage_client: StorageClient,
    worker_id: WorkerID,
    catch: Tuple[Type[Exception], ...],
    callbacks: Optional[Sequence[Callable[[Trial], None]]],
    trial_queue: _TrialQueue,
) -> None:
    trial = _ask(
        study_info=study_info,
        sampler=sampler,
        storage=storage_client,
        worker_id=worker_id,
        trial_queue=trial_queue,
    )
    try:
        values = objective(trial)
    except PrunedException:
        proto = trial.get_proto()
        proto.last_known_state = TrialProto.State.PRUNED
    except catch:
        proto = trial.get_proto()
        proto.last_known_state = TrialProto.State.FAILED
    else:
        proto = trial.get_proto()
        if isinstance(values, SequenceType):
            objective_values = [_value_to_objective_value(value=float(value)) for value in values]
        else:
            objective_values = [_value_to_objective_value(value=float(values))]
        del proto.values[:]
        proto.values.extend(objective_values)
        proto.last_known_state = _infer_trial_state_from_objective_values(objective_values)
    if callbacks:
        for callback in callbacks:
            # TODO(tsuzuku): Think about a better type to pass callbacks.
            callback(trial)
    storage_client.write_trial(trial=proto)


def _value_to_objective_value(value: float) -> ObjectiveValue:
    if math.isnan(value):
        return ObjectiveValue(status=ObjectiveValue.Status.NAN)
    elif math.isinf(value):
        if value > 0:
            return ObjectiveValue(status=ObjectiveValue.Status.INF, value=value)
        else:
            return ObjectiveValue(status=ObjectiveValue.Status.NEGATIVE_INF, value=value)
    else:
        return ObjectiveValue(status=ObjectiveValue.Status.VALID, value=value)


def _infer_trial_state_from_objective_values(
    values: Sequence[ObjectiveValue],
) -> "TrialProto.StateValue":
    if not values:
        return TrialProto.State.UNKNOWN
    if all(value.status == ObjectiveValue.Status.VALID for value in values):
        return TrialProto.State.COMPLETED
    if any(value.status == ObjectiveValue.Status.UNKNOWN for value in values):
        return TrialProto.State.UNKNOWN
    if any(
        value.status
        in (
            ObjectiveValue.Status.NAN,
            ObjectiveValue.INF,
            ObjectiveValue.NEGATIVE_INF,
            ObjectiveValue.INFEASIBLE,
        )
        for value in values
    ):
        return TrialProto.State.PARTIALLY_FAILED
    if all(
        value.status in (ObjectiveValue.Status.VALID, ObjectiveValue.SKIPPED) for value in values
    ):
        return TrialProto.State.PARTIALLY_COMPLETED
    return TrialProto.State.FAILED

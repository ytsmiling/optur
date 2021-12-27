import itertools
import math
from collections.abc import Sequence as SequenceType
from concurrent.futures import ThreadPoolExecutor
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
        if n_jobs > 1:
            # Storage instance cannot be shared by multiple threads
            # or processes. See :class:`~optur.storage.Storage`'s
            # classdoc for more details.
            clients = [self._storage.create_client() for _ in range(n_jobs)]
        else:
            # Avoid using storage's clients to reduce runtime overhead.
            # TODO(tsuzuku): Benchmark.
            clients = [self._storage]
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # TODO(tsuzuku): Use correct argument.s
            executor.map(_run_trials, [() for _ in clients], timeout=timeout)


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


class _TrialQueue:
    """Trial queue for managing WAITING trials."""

    def __init__(self, states: "Sequence[TrialProto.StateValue]") -> None:
        self._trials: Dict[str, TrialProto] = {}
        self._timestamp: Optional[Timestamp] = None
        self._states = states

    @property
    def last_update_time(self) -> Optional[Timestamp]:
        return self._timestamp

    def update_timestamp(self, timestamp: Optional[Timestamp]) -> None:
        self._timestamp = timestamp

    def sync(self, trials: Sequence[TrialProto]) -> None:
        for trial in trials:
            if trial.trial_id in self._trials:
                if trial.last_known_state in self._states:
                    self._trials[trial.trial_id] = trial
                else:
                    del self._trials[trial.trial_id]
            else:
                if trial.last_known_state in self._states:
                    self._trials[trial.trial_id] = trial

    def get_trial(self, state: "TrialProto.StateValue") -> Optional[TrialProto]:
        for trial in self._trials.values():
            if trial.last_known_state == state:
                del self._trials[trial.trial_id]
                return trial
        return None


def _ask(
    study_id: str,
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
    trials = storage.get_trials(study_id=study_id, timestamp=queue_timestamp)
    trial_queue.sync(trials)
    trial_queue.update_timestamp(new_timestamp)
    # TODO(tsuzuku): # Get waiting trial.
    initial_trial: Optional[TrialProto] = None
    assert initial_trial is not None
    # Sync sampler and storage.
    sampler_timestamp = sampler.last_update_time
    if queue_timestamp != sampler_timestamp:
        new_timestamp = storage.get_current_timestamp()
        trials = storage.get_trials(study_id=study_id, timestamp=sampler_timestamp)
    sampler.sync(trials)
    sampler.update_timestamp(new_timestamp)
    # Call joint_sample of sampler
    ret = Trial(trial_proto=initial_trial, storage=storage)
    # TODO(tsuzuku): Pass fixed_parameters and search_space.
    ret.update_parameters(sampler.joint_sample())
    return ret


def _run_trials(
    objective: ObjectiveFuncType,
    study_id: str,
    sampler_config: SamplerConfig,
    worker_id: WorkerID,
    storage_client: StorageClient,
    n_trials: Optional[int],
    catch: Tuple[Type[Exception], ...],
    callbacks: Optional[List[Callable[[Trial], None]]],
) -> None:
    # We need to create sampler instances per thread because
    # they are neither thread-safe nor process-safe.
    # Additionally, if we share sampler instances among workers,
    # we have risk that samplers' caches are updated during
    # trials' lifetime.
    # Unlike optuna, it is less likely that the update breaks
    # sampler algorithms in optur, but still, we want to ensure that samplers
    # see the same cache in all `joint_sample` and `sample` calls for the same trial.
    sampler = create_sampler(sampler_config=sampler_config)
    # Unlike samplers, we are free to share _TrialQueue between processes or threads.
    # We're creating the queue here because the current implementation of `_TrialQueue`
    # is not process/thread-safe.
    trial_queue = _TrialQueue([TrialProto.State.WAITING])
    trial_counter = itertools.count() if n_trials is None else range(n_trials)
    for _ in trial_counter:
        _run_trial(
            study_id=study_id,
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
    study_id: str,
    sampler: Sampler,
    storage_client: StorageClient,
    worker_id: WorkerID,
    catch: Tuple[Type[Exception], ...],
    callbacks: Optional[List[Callable[[Trial], None]]],
    trial_queue: _TrialQueue,
) -> None:
    trial = _ask(
        study_id=study_id,
        sampler=sampler,
        storage=storage_client,
        worker_id=worker_id,
        trial_queue=trial_queue,
    )
    try:
        values = objective(trial)
    except PrunedException:
        trial.proto.last_known_state = TrialProto.State.PRUNED
    except catch:
        trial.proto.last_known_state = TrialProto.State.FAILED
    else:
        if isinstance(values, SequenceType):
            objective_values = [_value_to_objective_value(value=float(value)) for value in values]
        else:
            objective_values = [_value_to_objective_value(value=float(values))]
        del trial.proto.values[:]
        trial.proto.values.extend(objective_values)
        trial.proto.last_known_state = _infer_trial_state_from_objective_values(objective_values)
    if callbacks:
        for callback in callbacks:
            callback(trial)
    trial.flush()


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

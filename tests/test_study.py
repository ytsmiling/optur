import math
import uuid
from unittest.mock import MagicMock, call

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.study_pb2 import ObjectiveValue, StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.proto.study_pb2 import WorkerID
from optur.study import (
    _ask,
    _infer_trial_state_from_objective_values,
    _value_to_objective_value,
)


def test_infer_trial_state_from_no_objective_values() -> None:
    assert _infer_trial_state_from_objective_values([]) == TrialProto.State.UNKNOWN


def test_infer_trial_state_from_unknown_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [ObjectiveValue(status=ObjectiveValue.Status.UNKNOWN)]
        )
        == TrialProto.State.UNKNOWN
    )
    assert (
        _infer_trial_state_from_objective_values(
            [ObjectiveValue(status=ObjectiveValue.Status.UNKNOWN) for _ in range(3)]
        )
        == TrialProto.State.UNKNOWN
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.INF),
                ObjectiveValue(status=ObjectiveValue.Status.UNKNOWN),
                ObjectiveValue(status=ObjectiveValue.Status.NEGATIVE_INF),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
            ]
        )
        == TrialProto.State.UNKNOWN
    )


def test_infer_trial_state_from_nan_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.NAN),
                ObjectiveValue(status=ObjectiveValue.Status.NAN),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.NAN),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )


def test_infer_trial_state_from_inf_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.INF),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.NEGATIVE_INF),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.INF),
                ObjectiveValue(status=ObjectiveValue.Status.NEGATIVE_INF),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )


def test_infer_trial_state_from_missing_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.SKIPPED),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_COMPLETED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.SKIPPED),
                ObjectiveValue(status=ObjectiveValue.Status.SKIPPED),
            ]
        )
        == TrialProto.State.PARTIALLY_COMPLETED
    )


def test_infer_trial_state_from_infeasible_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.INFEASIBLE),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.INFEASIBLE),
                ObjectiveValue(status=ObjectiveValue.Status.INFEASIBLE),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )


def test_nan_to_objective_value() -> None:
    assert _value_to_objective_value(math.nan).status == ObjectiveValue.Status.NAN


def test_inf_to_objective_value() -> None:
    assert _value_to_objective_value(math.inf).status == ObjectiveValue.Status.INF
    assert _value_to_objective_value(-math.inf).status == ObjectiveValue.Status.NEGATIVE_INF


def test_valid_value_to_objective_value() -> None:
    assert _value_to_objective_value(0.1).status == ObjectiveValue.Status.VALID
    assert math.isclose(_value_to_objective_value(0.2).value, 0.2)


def test_ask_sync_sampler() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    sampler_timestamp = Timestamp(seconds=1234)
    storage_timestamp = Timestamp(seconds=2345)
    queue_timestamp = Timestamp(seconds=3456)
    study_id = uuid.uuid4().hex
    trials = [TrialProto(trial_id=uuid.uuid4().hex)]
    sampler.last_update_time = sampler_timestamp
    storage.get_current_timestamp.return_value = storage_timestamp
    storage.get_trials.return_value = trials
    trial_queue = MagicMock()
    trial_queue.get_trial.return_value = None
    trial_queue.last_update_time = queue_timestamp
    _ask(
        study_info=StudyInfo(study_id=study_id),
        sampler=sampler,
        storage=storage,
        trial_queue=trial_queue,
        worker_id=WorkerID(),
    )
    assert any(
        call(study_id=study_id, timestamp=sampler_timestamp) == c
        for c in storage.get_trials.call_args_list
    )
    assert sampler.sync.call_args_list == [call(trials=trials)]
    assert sampler.update_timestamp.call_args_list == [call(timestamp=storage_timestamp)]


def test_ask_sync_trial_queue() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    sampler_timestamp = Timestamp(seconds=1234)
    storage_timestamp = Timestamp(seconds=2345)
    queue_timestamp = Timestamp(seconds=3456)
    study_id = uuid.uuid4().hex
    trials = [TrialProto(trial_id=uuid.uuid4().hex)]
    sampler.last_update_time = sampler_timestamp
    storage.get_current_timestamp.return_value = storage_timestamp
    storage.get_trials.return_value = trials
    trial_queue = MagicMock()
    trial_queue.get_trial.return_value = None
    trial_queue.last_update_time = queue_timestamp
    _ask(
        study_info=StudyInfo(study_id=study_id),
        sampler=sampler,
        storage=storage,
        trial_queue=trial_queue,
        worker_id=WorkerID(),
    )
    assert any(
        call(study_id=study_id, timestamp=queue_timestamp) == c
        for c in storage.get_trials.call_args_list
    )
    assert trial_queue.sync.call_args_list == [call(trials=trials)]
    assert trial_queue.update_timestamp.call_args_list == [call(timestamp=storage_timestamp)]


def test_ask_calls_joint_sample() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    sampler_timestamp = Timestamp(seconds=1234)
    storage_timestamp = Timestamp(seconds=2345)
    queue_timestamp = Timestamp(seconds=3456)
    study_id = uuid.uuid4().hex
    trials = [TrialProto(trial_id=uuid.uuid4().hex)]
    sampler.last_update_time = sampler_timestamp
    storage.get_current_timestamp.return_value = storage_timestamp
    storage.get_trials.return_value = trials
    trial_queue = MagicMock()
    trial_queue.get_trial.return_value = None
    trial_queue.last_update_time = queue_timestamp
    _ask(
        study_info=StudyInfo(study_id=study_id),
        sampler=sampler,
        storage=storage,
        trial_queue=trial_queue,
        worker_id=WorkerID(),
    )
    assert len(sampler.joint_sample.call_args_list) == 1


def test_ask_uses_waiting_trial() -> None:
    # TODO(tsuzuku): Test this.
    pass


def test_ask_sets_worker_id() -> None:
    # TODO(tsuzuku): Test this.
    pass

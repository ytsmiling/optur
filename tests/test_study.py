import math
import uuid
from typing import Any
from unittest.mock import MagicMock, call

import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from optur.errors import PrunedException
from optur.proto.sampler_pb2 import RandomSamplerConfig, SamplerConfig
from optur.proto.search_space_pb2 import ParameterValue
from optur.proto.study_pb2 import ObjectiveValue, StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.proto.study_pb2 import WorkerID
from optur.samplers.sampler import JointSampleResult
from optur.study import (
    _ask,
    _infer_trial_state_from_objective_values,
    _optimize,
    _run_trial,
    _run_trials,
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


def test_ask_sets_study_id() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    sampler_timestamp = Timestamp(seconds=1234)
    storage_timestamp = Timestamp(seconds=2345)
    queue_timestamp = Timestamp(seconds=3456)
    study_id = uuid.uuid4().hex
    trials = [TrialProto(trial_id=uuid.uuid4().hex)]
    sampler.last_update_time = sampler_timestamp
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
    storage.get_current_timestamp.return_value = storage_timestamp
    storage.get_trials.return_value = trials
    trial_queue = MagicMock()
    trial_queue.get_trial.return_value = None
    trial_queue.last_update_time = queue_timestamp
    trial = _ask(
        study_info=StudyInfo(study_id=study_id),
        sampler=sampler,
        storage=storage,
        trial_queue=trial_queue,
        worker_id=WorkerID(),
    )
    assert trial.get_proto().study_id == study_id


def test_ask_sync_sampler() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    sampler_timestamp = Timestamp(seconds=1234)
    storage_timestamp = Timestamp(seconds=2345)
    queue_timestamp = Timestamp(seconds=3456)
    study_id = uuid.uuid4().hex
    trials = [TrialProto(trial_id=uuid.uuid4().hex)]
    sampler.last_update_time = sampler_timestamp
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
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
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
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
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
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


def test_run_trial_uses_joint_sample() -> None:
    objective = MagicMock()
    sampler = MagicMock()
    storage = MagicMock()
    queue = MagicMock()
    objective.return_value = 0.1
    sampler.last_update_time = None
    sampler.joint_sample.return_value = JointSampleResult(
        parameters={
            "foo": ParameterValue(int_value=1),
            "bar": ParameterValue(double_value=2.0),
        },
        system_attrs={},
    )
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    queue.get_trial.return_value = None
    _run_trial(
        objective=objective,
        study_info=StudyInfo(),
        sampler=sampler,
        storage_client=storage,
        worker_id=WorkerID(),
        catch=(),
        callbacks=(),
        trial_queue=queue,
    )
    objective.assert_called_once
    args, _ = objective.call_args
    trial = args[0]
    assert trial.suggest_parameter("foo") == ParameterValue(int_value=1)
    assert trial.suggest_parameter("bar") == ParameterValue(double_value=2.0)


def test_run_trial_uses_waiting_trial() -> None:
    objective = MagicMock()
    sampler = MagicMock()
    storage = MagicMock()
    queue = MagicMock()
    objective.return_value = 0.1
    sampler.last_update_time = None
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    trial_id = uuid.uuid4().hex
    queue.get_trial.return_value = TrialProto(trial_id=trial_id)
    _run_trial(
        objective=objective,
        study_info=StudyInfo(),
        sampler=sampler,
        storage_client=storage,
        worker_id=WorkerID(),
        catch=(),
        callbacks=(),
        trial_queue=queue,
    )
    objective.assert_called_once
    args, _ = objective.call_args
    trial = args[0]
    assert trial.get_proto().trial_id == trial_id


@pytest.mark.parametrize(
    "values,state",
    [
        (0.1, TrialProto.State.COMPLETED),
        ([0.1, 0.2], TrialProto.State.COMPLETED),
        ([0.1, math.inf], TrialProto.State.PARTIALLY_FAILED),
        ([0.1, -math.inf], TrialProto.State.PARTIALLY_FAILED),
        ([math.inf, -math.inf], TrialProto.State.PARTIALLY_FAILED),
        ([math.nan, -math.inf], TrialProto.State.PARTIALLY_FAILED),
        ([math.nan, 0.2], TrialProto.State.PARTIALLY_FAILED),
    ],
)
def test_run_trial_return_value_handling(values: Any, state: "TrialProto.StateValue") -> None:
    objective = MagicMock()
    sampler = MagicMock()
    storage = MagicMock()
    queue = MagicMock()
    sampler.last_update_time = None
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    objective.return_value = values
    queue.get_trial.return_value = None
    _run_trial(
        objective=objective,
        study_info=StudyInfo(),
        sampler=sampler,
        storage_client=storage,
        worker_id=WorkerID(),
        catch=(),
        callbacks=(),
        trial_queue=queue,
    )
    storage.write_trial.assert_called_once
    _, kwargs = storage.write_trial.call_args
    trial = kwargs["trial"]
    assert trial.last_known_state == state


def test_run_trial_sets_pruned_state() -> None:
    objective = MagicMock()
    sampler = MagicMock()
    storage = MagicMock()
    queue = MagicMock()
    sampler.last_update_time = None
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    objective.side_effect = PrunedException
    queue.get_trial.return_value = None
    _run_trial(
        objective=objective,
        study_info=StudyInfo(),
        sampler=sampler,
        storage_client=storage,
        worker_id=WorkerID(),
        catch=(),
        callbacks=(),
        trial_queue=queue,
    )
    storage.write_trial.assert_called_once
    _, kwargs = storage.write_trial.call_args
    trial = kwargs["trial"]
    assert trial.last_known_state == TrialProto.State.PRUNED


def test_run_trial_catch_and_sets_failed_state() -> None:
    objective = MagicMock()
    sampler = MagicMock()
    storage = MagicMock()
    queue = MagicMock()
    sampler.last_update_time = None
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    objective.side_effect = RuntimeError
    queue.get_trial.return_value = None
    _run_trial(
        objective=objective,
        study_info=StudyInfo(),
        sampler=sampler,
        storage_client=storage,
        worker_id=WorkerID(),
        catch=(RuntimeError,),
        callbacks=(),
        trial_queue=queue,
    )
    storage.write_trial.assert_called_once
    _, kwargs = storage.write_trial.call_args
    trial = kwargs["trial"]
    assert trial.last_known_state == TrialProto.State.FAILED


def test_run_trial_does_not_catch() -> None:
    objective = MagicMock()
    sampler = MagicMock()
    storage = MagicMock()
    queue = MagicMock()
    sampler.last_update_time = None
    sampler.joint_sample.return_value = JointSampleResult(parameters={}, system_attrs={})
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    objective.side_effect = RuntimeError
    queue.get_trial.return_value = None
    with pytest.raises(RuntimeError):
        _run_trial(
            objective=objective,
            study_info=StudyInfo(),
            sampler=sampler,
            storage_client=storage,
            worker_id=WorkerID(),
            catch=(),
            callbacks=(),
            trial_queue=queue,
        )


def test_run_trials_call_objective_multiple_times() -> None:
    objective = MagicMock()
    storage = MagicMock()
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    _run_trials(
        objective=objective,
        study_info=StudyInfo(),
        sampler_config=SamplerConfig(random=RandomSamplerConfig()),
        worker_id=WorkerID(),
        storage_client=storage,
        n_trials=7,
        catch=(),
        callbacks=(),
    )
    assert len(objective.call_args_list) == 7


def test_optimize_single_worker_sanity_check() -> None:
    objective = MagicMock()
    storage = MagicMock()
    storage.get_current_timestamp.return_value = None
    storage.get_trials.return_value = []
    _optimize(
        objective=objective,
        study_info=StudyInfo(),
        sampler_config=SamplerConfig(random=RandomSamplerConfig()),
        client_id=uuid.uuid4().hex,
        storage=storage,
        n_trials=7,
        timeout=None,
        n_jobs=1,
        catch=(),
        callbacks=(),
    )
    if len(objective.call_args_list) != 7:
        if not objective.call_args_list:
            raise AssertionError(
                "Objective was not called. You might have changed the signature of _run_trials "
                "and `Executor` might have failed silently."
            )
        else:
            raise AssertionError(
                f"Objective was called {len(objective.call_args_list)} times, while "
                "7 times are requested."
            )

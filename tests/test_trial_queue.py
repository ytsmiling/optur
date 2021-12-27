import uuid

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.study_pb2 import Trial
from optur.study import _TrialQueue


def test_initial_timestamp_is_none() -> None:
    assert _TrialQueue(states=()).last_update_time is None


def test_update_timestamp() -> None:
    timestamp = Timestamp(seconds=42, nanos=123)
    queue = _TrialQueue(states=())
    queue.update_timestamp(timestamp)
    assert queue.last_update_time == timestamp


def test_enqueue_and_get_trial() -> None:
    queue = _TrialQueue(states=(Trial.State.WAITING,))
    assert queue.get_trial(state=Trial.State.WAITING) is None
    waiting_trials = [
        Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING),
        Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING),
    ]
    queue.sync(
        [
            Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.RUNNING),
        ]
        + waiting_trials
        + [
            Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.COMPLETED),
            Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.FAILED),
        ]
    )
    trials = [
        queue.get_trial(state=Trial.State.WAITING),
        queue.get_trial(state=Trial.State.WAITING),
    ]
    assert {trial.trial_id for trial in trials if trial is not None} == {
        trial.trial_id for trial in waiting_trials
    }
    assert queue.get_trial(state=Trial.State.WAITING) is None


def test_update_enqueued_trials() -> None:
    queue = _TrialQueue(states=(Trial.State.WAITING,))
    waiting_trials = [
        Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING),
        Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING),
        Trial(trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING),
    ]
    queue.sync(waiting_trials)
    queue.sync([Trial(trial_id=waiting_trials[1].trial_id, last_known_state=Trial.State.RUNNING)])
    trials = [
        queue.get_trial(state=Trial.State.WAITING),
        queue.get_trial(state=Trial.State.WAITING),
    ]
    assert {trial.trial_id for trial in trials if trial is not None} == {
        trial.trial_id for trial in [waiting_trials[0], waiting_trials[2]]
    }
    assert queue.get_trial(state=Trial.State.WAITING) is None

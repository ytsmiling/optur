import uuid

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.study_pb2 import Trial, WorkerID
from optur.study import _TrialQueue


def test_initial_timestamp_is_none() -> None:
    assert _TrialQueue(states=(), worker_id=WorkerID()).last_update_time is None


def test_update_timestamp() -> None:
    timestamp = Timestamp(seconds=42, nanos=123)
    queue = _TrialQueue(states=(), worker_id=WorkerID())
    queue.update_timestamp(timestamp)
    assert queue.last_update_time == timestamp


def test_enqueue_and_get_trial() -> None:
    worker_id = WorkerID(client_id=uuid.uuid4().hex)
    queue = _TrialQueue(states=(Trial.State.WAITING,), worker_id=worker_id)
    assert queue.get_trial(state=Trial.State.WAITING) is None
    waiting_trials = [
        Trial(
            trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING, worker_id=worker_id
        ),
        Trial(
            trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING, worker_id=worker_id
        ),
    ]
    queue.sync(
        [
            Trial(
                trial_id=uuid.uuid4().hex,
                last_known_state=Trial.State.RUNNING,
                worker_id=worker_id,
            ),
            Trial(
                trial_id=uuid.uuid4().hex,
                last_known_state=Trial.State.WAITING,
                worker_id=WorkerID(client_id=uuid.uuid4().hex),
            ),
        ]
        + waiting_trials
        + [
            Trial(
                trial_id=uuid.uuid4().hex,
                last_known_state=Trial.State.WAITING,
                worker_id=WorkerID(client_id=uuid.uuid4().hex),
            ),
            Trial(
                trial_id=uuid.uuid4().hex,
                last_known_state=Trial.State.COMPLETED,
                worker_id=worker_id,
            ),
            Trial(
                trial_id=uuid.uuid4().hex, last_known_state=Trial.State.FAILED, worker_id=worker_id
            ),
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
    worker_id = WorkerID()
    queue = _TrialQueue(states=(Trial.State.WAITING,), worker_id=worker_id)
    waiting_trials = [
        Trial(
            trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING, worker_id=worker_id
        ),
        Trial(
            trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING, worker_id=worker_id
        ),
        Trial(
            trial_id=uuid.uuid4().hex, last_known_state=Trial.State.WAITING, worker_id=worker_id
        ),
    ]
    queue.sync(waiting_trials)
    queue.sync(
        [
            Trial(
                trial_id=waiting_trials[1].trial_id,
                last_known_state=Trial.State.RUNNING,
                worker_id=worker_id,
            )
        ]
    )
    trials = [
        queue.get_trial(state=Trial.State.WAITING),
        queue.get_trial(state=Trial.State.WAITING),
    ]
    assert {trial.trial_id for trial in trials if trial is not None} == {
        trial.trial_id for trial in [waiting_trials[0], waiting_trials[2]]
    }
    assert queue.get_trial(state=Trial.State.WAITING) is None

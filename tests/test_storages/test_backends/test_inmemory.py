import random
import uuid

import pytest  # type: ignore

from optur.errors import NotFoundError
from optur.proto.study_pb2 import StudyInfo, Target, Trial
from optur.storages.backends.inmemory import InMemoryStorageBackend


def test_get_current_timestamp_is_ordered() -> None:
    backend = InMemoryStorageBackend()
    timestamps = [backend.get_current_timestamp() for _ in range(100)]
    int_timestamps = [(t.seconds, t.nanos) for t in timestamps]
    assert [t is not None for t in int_timestamps]
    sorted_timestamps = list(sorted(int_timestamps))
    assert int_timestamps == sorted_timestamps


def test_read_all_study() -> None:
    backend = InMemoryStorageBackend()
    # TODO(tsuzuku): Create `create_random_study_info` helper.
    studies = [
        StudyInfo(
            study_id=uuid.uuid4().hex,
            targets=[
                Target(
                    name=uuid.uuid4().urn,
                    direction=random.choice(
                        seq=[Target.Direction.MAXIMIZE, Target.Direction.MINIMIZE],
                    ),
                ),
            ],
        )
        for _ in range(13)
    ]
    for study in studies:
        backend.write_study(study=study)
    loaded_studies = backend.get_studies(timestamp=None)
    sorted_studies = list(sorted((s for s in studies), key=lambda x: x.study_id))
    sorted_loaded_studies = list(sorted((s for s in loaded_studies), key=lambda x: x.study_id))
    assert [(s.study_id, s.targets) for s in sorted_studies] == [
        (s.study_id, s.targets) for s in sorted_loaded_studies
    ]


def test_incremental_read_study() -> None:
    backend = InMemoryStorageBackend()
    # TODO(tsuzuku): Create `create_random_study_info` helper.
    studies = [
        StudyInfo(
            study_id=uuid.uuid4().hex,
            targets=[
                Target(
                    name=uuid.uuid4().urn,
                    direction=random.choice(
                        seq=[Target.Direction.MAXIMIZE, Target.Direction.MINIMIZE],
                    ),
                ),
            ],
        )
        for _ in range(13)
    ]
    for study in studies[:6]:
        backend.write_study(study=study)
    timestamp = backend.get_current_timestamp()
    for study in studies[6:]:
        backend.write_study(study=study)
    loaded_studies = [(s.study_id, s.targets) for s in backend.get_studies(timestamp=timestamp)]
    for study in studies[6:]:
        assert (study.study_id, study.targets) in loaded_studies


def test_read_write_trial() -> None:
    backend = InMemoryStorageBackend()
    study = StudyInfo(study_id=uuid.uuid4().hex)
    trial = Trial(trial_id=uuid.uuid4().hex, study_id=study.study_id, system_attrs={"foo": "bar"})
    backend.write_study(study=study)
    backend.write_trial(trial=trial)
    loaded_trial = backend.get_trial(trial_id=trial.trial_id)
    assert dict(loaded_trial.system_attrs.items()) == {"foo": "bar"}


def test_write_trial_with_non_existent_study() -> None:
    backend = InMemoryStorageBackend()
    study = StudyInfo(study_id=uuid.uuid4().hex)
    trial = Trial(
        trial_id=uuid.uuid4().hex, study_id=uuid.uuid4().hex, system_attrs={"foo": "bar"}
    )
    backend.write_study(study=study)
    with pytest.raises(NotFoundError):
        backend.write_trial(trial=trial)


def test_read_non_existent_trial() -> None:
    backend = InMemoryStorageBackend()
    study = StudyInfo(study_id=uuid.uuid4().hex)
    trial = Trial(trial_id=uuid.uuid4().hex, study_id=study.study_id, system_attrs={"foo": "bar"})
    backend.write_study(study=study)
    backend.write_trial(trial=trial)
    with pytest.raises(NotFoundError):
        backend.get_trial(trial_id=uuid.uuid4().hex)

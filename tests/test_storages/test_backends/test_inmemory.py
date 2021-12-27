import random
import uuid

from optur.proto.study_pb2 import StudyInfo, Target
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

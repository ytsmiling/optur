import os
import random
import uuid

import pytest

from optur.proto.study_pb2 import StudyInfo, Target
from optur.storages.backends.mysql import MySQLBackend


@pytest.mark.mysql
def test_init() -> None:
    backend = MySQLBackend(
        user=os.environ["MYSQL_USER"],
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", 3306)),
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
    )
    backend.init()


@pytest.mark.mysql
def test_get_current_timestamp_is_ordered() -> None:
    backend = MySQLBackend(
        user=os.environ["MYSQL_USER"],
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", 3306)),
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
    )
    timestamps = [backend.get_current_timestamp() for _ in range(10)]
    int_timestamps = [(t.seconds, t.nanos) for t in timestamps]
    assert [t is not None for t in int_timestamps]
    sorted_timestamps = list(sorted(int_timestamps))
    assert int_timestamps == sorted_timestamps


@pytest.mark.mysql
def test_read_all_study() -> None:
    backend = MySQLBackend(
        user=os.environ["MYSQL_USER"],
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", 3306)),
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
    )
    backend.init()
    backend.drop_all()
    backend.init()
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

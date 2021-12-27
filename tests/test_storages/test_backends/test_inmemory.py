from optur.storages.backends.inmemory import InMemoryStorageBackend


def test_get_current_timestamp_is_ordered() -> None:
    backend = InMemoryStorageBackend()
    timestamps = [backend.get_current_timestamp() for _ in range(100)]
    int_timestamps = [(t.seconds, t.nanos) for t in timestamps]
    assert [t is not None for t in int_timestamps]
    sorted_timestamps = list(sorted(int_timestamps))
    assert int_timestamps == sorted_timestamps

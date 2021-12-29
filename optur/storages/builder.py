from optur.storages.backends.inmemory import InMemoryStorageBackend
from optur.storages.storage import Storage


def create_inmemory_storage() -> Storage:
    return Storage(backend=InMemoryStorageBackend())

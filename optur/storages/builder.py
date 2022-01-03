import pathlib
from typing import Union

from optur.storages.backends.inmemory import InMemoryStorageBackend
from optur.storages.backends.posix import PosixStorageBackend
from optur.storages.storage import Storage


def create_inmemory_storage() -> Storage:
    return Storage(backend=InMemoryStorageBackend())


def create_posix_storage(root_dir: Union[str, pathlib.Path]) -> Storage:
    return Storage(backend=PosixStorageBackend(root_dir=root_dir))

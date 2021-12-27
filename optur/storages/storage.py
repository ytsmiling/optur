import abc
from typing import List, Optional

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.study_pb2 import StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.storages.backends.backend import StorageBackend


class StorageClient(abc.ABC):
    @abc.abstractclassmethod
    def get_current_timestamp(self) -> Timestamp:
        """Get current server-timestamp.

        For incrementally loading trials and studies, some server methods
        rely on timestamp.
        Since it's difficult to ensure that clocks are synchronized across workers,
        this method ask the current timestamp to the storage's backend and returns the response.

        Returns:
            Current server-timestamp (time since epoch).
        """
        pass

    @abc.abstractclassmethod
    def get_studies(self, timestamp: Optional[Timestamp] = None) -> List[StudyInfo]:
        """Get studies from the storage.

        Studies updated on or after the timestamp must be fetched.
        When the timestamp is :obj:`None`, all studies will be fetched.
        When the timestamp is set, only studies updated on or after the timestamp
        are guaranteed to be fetched.
        For studies updated before the timestamp, this method may or may not include
        them to the return value.

        Use the timestamp argument to incrementally load studies.

        Args:
            timestamp:
                Time from epoch.

        Returns:
            A list of :class:`~optur.proto.study_pb2.StudyInfo`.
        """
        pass

    @abc.abstractclassmethod
    def get_trials(
        self, study_id: Optional[str] = None, timestamp: Optional[Timestamp] = None
    ) -> List[TrialProto]:
        """Get trials from the storage.

        Studies updated on or after the timestamp must be fetched.
        When the timestamp is :obj:`None`, all trials will be fetched.
        When the timestamp is set, only trials updated on or after the timestamp
        are guaranteed to be fetched.
        For trials updated before the timestamp, this method may or may not include
        them to the return value.

        Use the timestamp argument to incrementally load trials.

        Args:
            study_id:
                ID of the study.
                Trials in all studies will be fetched when this argument is :obj:`None`.
            timestamp:
                Time from epoch.

        Returns:
            A list of :class:`~optur.proto.study_pb2.Trial`.
        """
        pass

    @abc.abstractclassmethod
    def get_trial(self, trial_id: str, study_id: Optional[str] = None) -> TrialProto:
        """Read a trial from the storage.

        Args:
            trial_id:
                ID of the trial.
            study_id:
                ID of the study to which the trial belongs.
                The storage may or may not use the information to speed up the lookup
                when available.

        Returns:
            The :class:`~optur.proto.study_pb2.Trial`.
        """
        pass

    @abc.abstractclassmethod
    def write_study(self, study: StudyInfo) -> None:
        """Write :class:`~optur.proto.study_pb2.StudyInfo` to the storage.

        This method overwrites existing study.
        Write operations might be out-of-order, but write operations by the
        same worker will always be the expected order.

        Args:
            study:
                A :class:`~optur.proto.study_pb2.StudyInfo` to write.
        """
        pass

    @abc.abstractclassmethod
    def write_trial(self, trial: TrialProto) -> None:
        """Write :class:`~optur.proto.study_pb2.Trial` to the storage.

        This method overwrites existing study.
        Write operations might be out-of-order, but write operations by the
        same worker will always be the expected order.

        Args:
            trial:
                A :class:`~optur.proto.study_pb2.Trial` to write.
        """
        pass


class Storage(StorageClient):
    """Storage class that has a StorageBackend.

    This class is permitted to have `StorageBackend`, but this introduces
    several restrictions to this class.
    First, this class must not be forked. This is because some `StorageBackend`s might
    not be picklable. This also helps avoiding wasting resources like connections to DB.
    Second, this class must not be used from multiple thread concurrently. This is because
    some `StorageBackend` might not be thread-safe. It is allowed to share the same instance
    with multiple threads, but its user's responsibility that ensures no concurrent access.

    To circumvent the restrictions, this class provides `create_trial` method.
    This method creates a `StorageClient` instance that communicates to the storage backend
    via this class.
    """

    def __init__(self, backend: StorageBackend) -> None:
        super().__init__()
        self._backend = backend

    def get_current_timestamp(self) -> Timestamp:
        return self._backend.get_current_timestamp()

    def get_studies(self, timestamp: Optional[Timestamp] = None) -> List[StudyInfo]:
        return self._backend.get_studies(timestamp=timestamp)

    def get_trials(
        self, study_id: Optional[str] = None, timestamp: Optional[Timestamp] = None
    ) -> List[TrialProto]:
        return self._backend.get_trials(study_id=study_id, timestamp=timestamp)

    def get_trial(self, trial_id: str, study_id: Optional[str] = None) -> TrialProto:
        return self._backend.get_trial(trial_id=trial_id, study_id=study_id)

    def write_study(self, study: StudyInfo) -> None:
        return self._backend.write_study(study=study)

    def write_trial(self, trial: TrialProto) -> None:
        return self._backend.write_trial(trial=trial)

    def create_client(self) -> StorageClient:
        raise NotImplementedError()

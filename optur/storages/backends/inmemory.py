import bisect
import datetime
from typing import Dict, List, NamedTuple, Optional

from google.protobuf.timestamp_pb2 import Timestamp

from optur.proto.study_pb2 import StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.storages.backends.backend import StorageBackend


# This class is used to provide a total order of trials.
class _TrialData(NamedTuple):
    # Timestamp converted into int.
    timestamp: int
    # To avoid direct comparison of `TrialProto`, which is undefined,
    # we add a unique field.
    trial_id: str
    trial: TrialProto


def _timestamp_to_int(t: Timestamp) -> int:
    return t.seconds * 10 ** 9 + t.nanos


class _StudyData(NamedTuple):
    study_info: StudyInfo
    # Large sorted list of finished trials.
    # These trials are sorted by ``last_update_time``.
    # Since ``key`` argument of ``bisect.bisect_left`` is not supported in python<=3.9,
    # this field has type that have total order.
    sorted_finished_trials: List[_TrialData]
    # Frequently updated but small.
    running_trials: List[TrialProto]


# This class is not thread-safe, and it's okay.
# This class is not process-safe, and it's okay.
class InMemoryStorageBackend(StorageBackend):
    def __init__(self) -> None:
        super().__init__()
        # Mapping from study_id to studies.
        self._studies: Dict[str, _StudyData] = {}
        # Mapping from trial_id to trials.
        self._trials: Dict[str, TrialProto] = {}

    def get_current_timestamp(self) -> Timestamp:
        timestamp = Timestamp()
        timestamp.FromDatetime(datetime.datetime.now())
        return timestamp

    def get_studies(self, timestamp: Optional[Timestamp] = None) -> List[StudyInfo]:
        """Get studies from the storage.

        Studies updated on or after the timestamp must be fetched.
        When the timestamp is :obj:`None`, all studies will be fetched.
        When the timestamp is set, only studies updated on or after the timestamp
        are guaranteed to be fetched.
        For studies updated before the timestamp, this method may or may not include
        them to the return value.

        Args:
            timestamp:
                Time from epoch.

        Returns:
            A list of :class:`~optur.proto.study_pb2.StudyInfo`.
        """

        return [study.study_info for study in self._studies.values()]

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

        if study_id is None:
            # TODO(tsuzuku): Support this.
            raise NotImplementedError()
        study = self._studies[study_id]
        finished_trials = study.sorted_finished_trials
        running_trials = study.running_trials
        if timestamp is None:
            ret = running_trials.copy()
            ret.extend((trial.trial for trial in finished_trials))
            return ret
        ret = running_trials.copy()
        left_idx = bisect.bisect_left(
            finished_trials, (_timestamp_to_int(timestamp) - 1, "", None)
        )
        ret.extend((trial.trial for trial in finished_trials[left_idx:]))
        return ret

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

        return self._trials[trial_id]

    def write_study(self, study: StudyInfo) -> None:
        """Write :class:`~optur.proto.study_pb2.StudyInfo` to the storage.

        This method overwrites existing study.
        Write operations might be out-of-order, but write operations by the
        same worker will always be the expected order.

        Args:
            study:
                A :class:`~optur.proto.study_pb2.StudyInfo` to write.
        """
        new_study = StudyInfo()
        new_study.CopyFrom(study)
        new_study.last_update_time.CopyFrom(self.get_current_timestamp())
        if study.study_id in self._studies:
            self._studies[study.study_id] = self._studies[study.study_id]._replace(
                study_info=new_study
            )
        else:
            self._studies[study.study_id] = _StudyData(
                study_info=new_study,
                sorted_finished_trials=[],
                running_trials=[],
            )

    def write_trial(self, trial: TrialProto) -> None:
        """Write :class:`~optur.proto.study_pb2.Trial` to the storage.

        This method overwrites existing study.
        Write operations might be out-of-order, but write operations by the
        same worker will always be the expected order.

        Args:
            trial:
                A :class:`~optur.proto.study_pb2.Trial` to write.
        """
        # TODO(tsuzuku): Handle `KeyError`.
        study = self._studies[trial.study_id]
        new_trial = TrialProto()
        new_trial.CopyFrom(trial)
        new_trial.last_update_time.CopyFrom(self.get_current_timestamp())
        old_trial = self._trials.get(trial.trial_id)
        self._trials[trial.trial_id] = new_trial
        # Delete an existing trial.
        if old_trial is not None:
            if old_trial.last_known_state in (TrialProto.State.RUNNING,):
                # TODO(tsuzuku): Find trial from self._studies[trial.study_id].running_trials.
                # TODO(tsuzuku): Update in-place if possible.
                raise NotImplementedError()
            elif old_trial.last_known_state in (
                TrialProto.State.COMPLETED,
                TrialProto.State.PARTIALLY_COMPLETED,
                TrialProto.State.PARTIALLY_FAILED,
                TrialProto.State.FAILED,
                TrialProto.State.PRUNED,
            ):
                # TODO(tsuzuku): Find trial from self._studies[trial.study_id].finished_trials.
                raise NotImplementedError()
            else:
                # TODO(tsuzuku): Think about UNKNOWNN, CREATED, and WAITING.
                raise NotImplementedError()
        # Add a new trial.
        if new_trial.last_known_state == TrialProto.State.RUNNING:
            study.running_trials.append(new_trial)
        elif new_trial.last_known_state in (
            TrialProto.State.COMPLETED,
            TrialProto.State.PARTIALLY_COMPLETED,
            TrialProto.State.PARTIALLY_FAILED,
            TrialProto.State.FAILED,
            TrialProto.State.PRUNED,
        ):
            study.sorted_finished_trials.append(
                _TrialData(
                    timestamp=_timestamp_to_int(new_trial.last_update_time),
                    trial_id=new_trial.trial_id,
                    trial=new_trial,
                )
            )
        else:
            # TODO(tsuzuku): Think about UNKNOWNN, CREATED, and WAITING.
            raise NotImplementedError()

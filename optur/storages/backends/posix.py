import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Union

from google.protobuf.timestamp_pb2 import Timestamp

from optur.errors import NotFoundError
from optur.proto.study_pb2 import StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.storages.backends.backend import StorageBackend


# The directory structure will be like
# ```
# root_dir/
#   .tmp/
#   optur_study_{study_id}/
#     study_info.pb
#     trial_{trial_id}.pb
#     trial_{trial_id}.pb
#   optur_study_{study_id}/
# ```
class PosixStorageBackend(StorageBackend):
    def __init__(self, root_dir: Union[str, Path]) -> None:
        super().__init__()
        self._root_dir = Path(root_dir)
        if not self._root_dir.is_dir():
            raise ValueError("")  # TODO(tsuzuku)
        # Do not use `tmpfile.TemporaryDirectory` because this class tries
        # to move files in the ``_tmpdir`` to ``_root_dir``.
        # When `tmpfile.TemporaryDirectory` and ``_root_dir`` are mounted
        # on a different storages, the move operation fails.
        self._tmpdir = self._root_dir / ".tmp"
        self._tmpdir.mkdir(exist_ok=True)

    def get_current_timestamp(self) -> Optional[Timestamp]:
        return None

    def _get_study_dir(self, study_id: str) -> Path:
        return self._root_dir / f"optur_study_{study_id}"

    def _get_study_file(self, study_dir: Path) -> Path:
        ret: Path = study_dir / "study_info.pb"
        return ret

    def _get_trial_file(self, study_dir: Path, trial_id: str) -> Path:
        ret: Path = study_dir / f"trial_{trial_id}.pb"
        return ret

    def get_studies(self, timestamp: Optional[Timestamp] = None) -> List[StudyInfo]:
        ret: List[StudyInfo] = []
        for directory in self._root_dir.glob("optur_study_*"):
            if not directory.is_dir():
                continue
            info_file = self._get_study_file(study_dir=directory)
            if not info_file.is_file():
                continue
            with info_file.open("rb") as f:
                ret.append(StudyInfo.FromString(f.read()))
        return ret

    def get_trials(
        self, study_id: Optional[str] = None, timestamp: Optional[Timestamp] = None
    ) -> List[TrialProto]:
        # Note, this method is "atomic".
        # TODO(tsuzuku): Support timestamp (as an optional feature).
        if study_id is None:
            ret: List[TrialProto] = []
            for study_dir in self._root_dir.glob("optur_study_*"):
                if not study_dir.is_dir():
                    continue
                ret.extend(self._get_trials(study_dir=study_dir, timestamp=timestamp))
            return ret
        study_dir = self._get_study_dir(study_id=study_id)
        if not study_dir.is_dir():
            raise NotFoundError("")  # TODO(tsuzuku)
        return self._get_trials(study_dir=study_dir, timestamp=timestamp)

    def _get_trials(self, study_dir: Path, timestamp: Optional[Timestamp]) -> List[TrialProto]:
        ret: List[TrialProto] = []
        for trial_file in study_dir.glob("trial_*.pb"):
            if not trial_file.is_file():
                pass
            with trial_file.open("rb") as f:
                ret.append(TrialProto.FromString(f.read()))
        return ret

    def get_trial(self, trial_id: str, study_id: Optional[str] = None) -> TrialProto:
        if study_id is None:
            for directory in self._root_dir.glob("optur_study_*"):
                if not directory.is_dir():
                    continue
                ret = self._get_trial(trial_id=trial_id, study_dir=directory)
                if ret is not None:
                    return ret
            raise NotFoundError("")  # TODO(tsuzuku)
        study_dir = self._get_study_dir(study_id=study_id)
        if not study_dir.is_dir():
            raise NotFoundError("")  # TODO(tsuzuku)
        ret = self._get_trial(trial_id=trial_id, study_dir=study_dir)
        if ret is None:
            raise NotFoundError("")  # TODO(tsuzuku)
        return ret

    def _get_trial(self, trial_id: str, study_dir: Path) -> Optional[TrialProto]:
        trial_path = self._get_trial_file(study_dir=study_dir, trial_id=trial_id)
        if trial_path.is_file():
            with trial_path.open("rb") as f:
                return TrialProto.FromString(f.read())
        return None

    def write_study(self, study: StudyInfo) -> None:
        study_dir = self._get_study_dir(study_id=study.study_id)
        study_dir.mkdir(exist_ok=True)
        study_file = self._get_study_file(study_dir=study_dir)
        tmpfile = self._tmpdir / uuid.uuid4().hex
        with tmpfile.open("wb") as f:
            f.write(study.SerializeToString())
        shutil.move(src=str(tmpfile), dst=study_file)

    def write_trial(self, trial: TrialProto) -> None:
        study_dir = self._get_study_dir(study_id=trial.study_id)
        if not study_dir.is_dir():
            raise NotFoundError("")  # TODO(tsuzuku)
        trial_file = self._get_trial_file(study_dir=study_dir, trial_id=trial.trial_id)
        tmpfile = self._tmpdir / uuid.uuid4().hex
        with tmpfile.open("wb") as f:
            f.write(trial.SerializeToString())
        shutil.move(src=str(tmpfile), dst=trial_file)

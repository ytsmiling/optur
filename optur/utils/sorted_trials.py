import math
from typing import Callable, List, Optional, Sequence

from optur.proto.study_pb2 import Target, Trial


class TrialFilter:
    def __init__(self, *, filter_unknown: bool = True) -> None:
        self._filter_unknown = filter_unknown

    def __call__(self, trial: Trial) -> bool:
        if self._filter_unknown and trial.last_known_state == Trial.State.UNKNOWN:
            return False
        return True


# Smaller is better.
class TrialKeyGenerator:
    def __init__(self, targets: Sequence[Target]) -> None:
        self._targets = targets

    @property
    def is_valid(self) -> bool:
        pass

    def __call__(self, trial: Trial) -> float:
        if trial.last_known_state != Trial.State.COMPLETED:
            return math.inf
        if self._targets[0].direction == Target.Direction.MINIMIZE:
            return trial.values[0].value
        else:
            return -trial.values[0].value


# LQ, not LE.
# Fail < Partially Failed < Pruned < (Partially Complete, Complete)
class TrialComparator:
    LQ_STATE = {
        (Trial.State.FAILED, Trial.State.PARTIALLY_FAILED),
        (Trial.State.FAILED, Trial.State.PRUNED),
        (Trial.State.FAILED, Trial.State.PARTIALLY_COMPLETED),
        (Trial.State.FAILED, Trial.State.COMPLETED),
        (Trial.State.PARTIALLY_FAILED, Trial.State.PRUNED),
        (Trial.State.PARTIALLY_FAILED, Trial.State.PARTIALLY_COMPLETED),
        (Trial.State.PARTIALLY_FAILED, Trial.State.COMPLETED),
        (Trial.State.PRUNED, Trial.State.PARTIALLY_COMPLETED),
        (Trial.State.PRUNED, Trial.State.COMPLETED),
    }

    def __init__(self, targets: Sequence[Target]) -> None:
        self._targets = targets

    def __call__(self, a: Trial, b: Trial) -> bool:
        if Trial.State.UNKNOWN in (a.last_known_state, b.last_known_state):
            raise RuntimeError()
        if a.last_known_state == b.last_known_state:
            if a.last_known_state == Trial.State.COMPLETED:
                raise NotImplementedError()
            elif a.last_known_state == Trial.State.PARTIALLY_COMPLETED:
                raise NotImplementedError()
            return False
        if (a.last_known_state, b.last_known_state) in self.LQ_STATE:
            return True
        # TODO(tsuzuku): Handle complete & Partially complete.
        return False


class SortedTrials:
    def __init__(
        self,
        trial_filter: Callable[[Trial], bool],
        trial_key_generator: Optional[Callable[[Trial], float]],
        trial_comparator: Optional[Callable[[Trial, Trial], bool]],
    ) -> None:
        assert trial_key_generator is not None or trial_comparator is not None
        self._trial_filter = trial_filter
        self._trial_key_generator = trial_key_generator
        self._trial_comparator = trial_comparator
        self._sorted_trials: List[Trial] = []
        if self._trial_key_generator is None:
            raise NotImplementedError("Multi-objective is not supported yet.")

    def sync(self, trials: Sequence[Trial]) -> None:
        """Update an internal data structure using the trials.

        When there are duplicated trials, new trials replace old ones.

        Let M be the number of trials and N be the length of this list before the sync.
        In single-objective study, this operation takes O(Mlog(M) + N).
        In multi-objective study, this operation takes O(M(M + N)).
        """
        if not trials:
            return
        assert self._trial_key_generator is not None, "Only single objective is supported."
        sorted_trials = list(
            sorted(filter(self._trial_filter, trials), key=self._trial_key_generator)
        )
        new_trials = {trial.trial_id for trial in sorted_trials}
        old_trials = [trial for trial in self._sorted_trials if trial.trial_id not in new_trials]
        self._sorted_trials = self._merge_sorted_trials(
            old_trials, sorted_trials, self._trial_key_generator
        )

    @staticmethod
    def _merge_sorted_trials(
        a: Sequence[Trial], b: Sequence[Trial], trial_key_generator: Callable[[Trial], float]
    ) -> List[Trial]:
        ret: List[Trial] = []
        a_idx = 0
        b_idx = 0
        while a_idx < len(a) and b_idx < len(b):
            if trial_key_generator(a[a_idx]) < trial_key_generator(b[b_idx]):
                ret.append(a[a_idx])
                a_idx += 1
            else:
                ret.append(b[b_idx])
                b_idx += 1
        if a_idx == len(a):
            ret.extend(b[b_idx:])
        else:
            ret.extend(a[a_idx:])
        return ret

    def to_list(self) -> List[Trial]:
        """Convert trials into a list.

        Let N be the number of stored trials. Then, this operation takes at most O(N).
        """
        assert self._trial_key_generator is not None, "Only single objective is supported."
        return self._sorted_trials

    def n_trials(self) -> int:
        """The number of stored trials."""
        assert self._trial_key_generator is not None, "Only single objective is supported."
        return len(self._sorted_trials)

    def get_best_trials(self) -> List[Trial]:
        pass

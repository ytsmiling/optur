from typing import List, Sequence

from optur.proto.study_pb2 import Trial


class SortedTrials:
    def __init__(self) -> None:
        pass

    def sync(self, trials: Sequence[Trial]) -> None:
        """Update an internal data structure using the trials.

        When there are duplicated trials, new trials replace old ones.

        Let M be the number of trials and N be the length of this list before the sync.
        In single-objective study, this operation takes O(Mlog(M) + N).
        In multi-objective study, this operation takes O(MN).
        """
        pass

    def to_list(self) -> List[Trial]:
        """Convert trials into a list.

        Let N be the number of stored trials. Then, this operation takes O(N).
        """
        pass

    def get_best_trials(self) -> List[Trial]:
        pass

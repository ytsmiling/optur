from optur.proto.study_pb2 import Trial

from optur.utils.sorted_trials import (  # SortedTrials,; TrialComparator,; TrialKeyGenerator,
    TrialQualityFilter,
)


def test_trial_quality_filter_remove_unknown() -> None:
    assert not TrialQualityFilter(filter_unknown=True)(Trial(last_known_state=Trial.State.UNKNOWN))
    assert TrialQualityFilter(filter_unknown=True)(Trial(last_known_state=Trial.State.FAILED))
    assert TrialQualityFilter(filter_unknown=True)(
        Trial(last_known_state=Trial.State.PARTIALLY_FAILED)
    )
    assert TrialQualityFilter(filter_unknown=False)(Trial(last_known_state=Trial.State.UNKNOWN))

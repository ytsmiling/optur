import math
import uuid

from optur.proto.study_pb2 import ObjectiveValue, Target, Trial
from optur.utils.sorted_trials import (
    SortedTrials,
    TrialKeyGenerator,
    TrialQualityFilter,
)


def test_trial_quality_filter_remove_unknown() -> None:
    assert not TrialQualityFilter(filter_unknown=True)(Trial(last_known_state=Trial.State.UNKNOWN))
    assert TrialQualityFilter(filter_unknown=True)(Trial(last_known_state=Trial.State.FAILED))
    assert TrialQualityFilter(filter_unknown=True)(
        Trial(last_known_state=Trial.State.PARTIALLY_FAILED)
    )
    assert TrialQualityFilter(filter_unknown=False)(Trial(last_known_state=Trial.State.UNKNOWN))


def test_trial_key_generator_is_not_valid_when_multiobjective() -> None:
    assert TrialKeyGenerator(targets=[Target(direction=Target.Direction.MAXIMIZE)]).is_valid
    assert TrialKeyGenerator(targets=[Target(direction=Target.Direction.MINIMIZE)]).is_valid
    assert not TrialKeyGenerator(targets=[Target(direction=Target.Direction.UNKNOWN)]).is_valid
    assert not TrialKeyGenerator(
        targets=[
            Target(direction=Target.Direction.MINIMIZE),
            Target(direction=Target.Direction.MAXIMIZE),
        ]
    ).is_valid
    assert TrialKeyGenerator(
        targets=[
            Target(direction=Target.Direction.MINIMIZE),
            Target(direction=Target.Direction.UNKNOWN),
        ]
    ).is_valid


def test_trial_key_generator_returns_inf_when_failed() -> None:
    generator = TrialKeyGenerator(targets=[Target(direction=Target.Direction.MAXIMIZE)])
    assert math.isinf(generator(Trial(last_known_state=Trial.State.UNKNOWN)))
    assert math.isinf(generator(Trial(last_known_state=Trial.State.PARTIALLY_FAILED)))
    # TODO(tsuzuku): We may check the value when it's PARTIALLY_COMPLETED.
    assert math.isinf(generator(Trial(last_known_state=Trial.State.PARTIALLY_COMPLETED)))


def test_trial_key_generator_returns_the_return_value_when_completed() -> None:
    generator = TrialKeyGenerator(targets=[Target(direction=Target.Direction.MAXIMIZE)])
    assert math.isclose(
        generator(
            Trial(
                last_known_state=Trial.State.COMPLETED,
                values=[ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.3)],
            )
        ),
        -0.3,
    )
    assert math.isclose(
        generator(
            Trial(
                last_known_state=Trial.State.COMPLETED,
                values=[ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.8)],
            )
        ),
        -0.8,
    )
    generator = TrialKeyGenerator(targets=[Target(direction=Target.Direction.MINIMIZE)])
    assert math.isclose(
        generator(
            Trial(
                last_known_state=Trial.State.COMPLETED,
                values=[ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.3)],
            )
        ),
        0.3,
    )
    assert math.isclose(
        generator(
            Trial(
                last_known_state=Trial.State.COMPLETED,
                values=[ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.8)],
            )
        ),
        0.8,
    )
    generator = TrialKeyGenerator(
        targets=[
            Target(direction=Target.Direction.UNKNOWN),
            Target(direction=Target.Direction.MINIMIZE),
        ]
    )
    assert math.isclose(
        generator(
            Trial(
                last_known_state=Trial.State.COMPLETED,
                values=[
                    ObjectiveValue(),
                    ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.3),
                ],
            )
        ),
        0.3,
    )
    assert math.isclose(
        generator(
            Trial(
                last_known_state=Trial.State.COMPLETED,
                values=[
                    ObjectiveValue(),
                    ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.8),
                ],
            )
        ),
        0.8,
    )


def test_sorted_trials_sort_all_trials() -> None:
    sorted_trials = SortedTrials(
        trial_filter=lambda t: True,
        trial_key_generator=lambda t: uuid.UUID(hex=t.trial_id).int,
        trial_comparator=None,
    )
    trials = [Trial(trial_id=uuid.uuid4().hex) for _ in range(100)]
    trials2 = list(sorted(trials, key=lambda t: uuid.UUID(hex=t.trial_id).int))
    sorted_trials.sync(trials=trials)
    assert sorted_trials.to_list() == trials2


def test_sorted_trials_filter_trials() -> None:
    trials = [Trial(trial_id=uuid.uuid4().hex) for _ in range(100)]
    sorted_trials = SortedTrials(
        trial_filter=lambda t: t.trial_id not in {x.trial_id for x in trials[30:50]},
        trial_key_generator=lambda t: uuid.UUID(hex=t.trial_id).int,
        trial_comparator=None,
    )
    trials2 = list(sorted(trials[:30] + trials[50:], key=lambda t: uuid.UUID(hex=t.trial_id).int))
    sorted_trials.sync(trials=trials)
    assert sorted_trials.to_list() == trials2

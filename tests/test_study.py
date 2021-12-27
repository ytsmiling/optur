import math

from optur.proto.study_pb2 import ObjectiveValue
from optur.proto.study_pb2 import Trial as TrialProto
from optur.study import (
    _infer_trial_state_from_objective_values,
    _value_to_objective_value,
)


def test_infer_trial_state_from_no_objective_values() -> None:
    assert _infer_trial_state_from_objective_values([]) == TrialProto.State.UNKNOWN


def test_infer_trial_state_from_unknown_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [ObjectiveValue(status=ObjectiveValue.Status.UNKNOWN)]
        )
        == TrialProto.State.UNKNOWN
    )
    assert (
        _infer_trial_state_from_objective_values(
            [ObjectiveValue(status=ObjectiveValue.Status.UNKNOWN) for _ in range(3)]
        )
        == TrialProto.State.UNKNOWN
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.INF),
                ObjectiveValue(status=ObjectiveValue.Status.UNKNOWN),
                ObjectiveValue(status=ObjectiveValue.Status.NEGATIVE_INF),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
            ]
        )
        == TrialProto.State.UNKNOWN
    )


def test_infer_trial_state_from_nan_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.NAN),
                ObjectiveValue(status=ObjectiveValue.Status.NAN),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.NAN),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )


def test_infer_trial_state_from_inf_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.INF),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.NEGATIVE_INF),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.INF),
                ObjectiveValue(status=ObjectiveValue.Status.NEGATIVE_INF),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )


def test_infer_trial_state_from_missing_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.SKIPPED),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_COMPLETED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.SKIPPED),
                ObjectiveValue(status=ObjectiveValue.Status.SKIPPED),
            ]
        )
        == TrialProto.State.PARTIALLY_COMPLETED
    )


def test_infer_trial_state_from_infeasible_objective_values() -> None:
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.1),
                ObjectiveValue(status=ObjectiveValue.Status.INFEASIBLE),
                ObjectiveValue(status=ObjectiveValue.Status.VALID, value=0.2),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )
    assert (
        _infer_trial_state_from_objective_values(
            [
                ObjectiveValue(status=ObjectiveValue.Status.INFEASIBLE),
                ObjectiveValue(status=ObjectiveValue.Status.INFEASIBLE),
            ]
        )
        == TrialProto.State.PARTIALLY_FAILED
    )


def test_nan_to_objective_value() -> None:
    assert _value_to_objective_value(math.nan).status == ObjectiveValue.Status.NAN


def test_inf_to_objective_value() -> None:
    assert _value_to_objective_value(math.inf).status == ObjectiveValue.Status.INF
    assert _value_to_objective_value(-math.inf).status == ObjectiveValue.Status.NEGATIVE_INF


def test_valid_value_to_objective_value() -> None:
    assert _value_to_objective_value(0.1).status == ObjectiveValue.Status.VALID
    assert math.isclose(_value_to_objective_value(0.2).value, 0.2)

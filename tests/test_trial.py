from unittest.mock import MagicMock, call

from optur.proto.search_space_pb2 import Distribution, ParameterValue
from optur.proto.study_pb2 import Parameter, StudyInfo
from optur.proto.study_pb2 import Trial as TrialProto
from optur.trial import Trial


def test_init_trial_does_not_use_sampler_or_storage() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    _ = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    assert sampler.mock_calls == []
    assert storage.mock_calls == []


def test_suggest_parameter_use_sampler_when_not_suggested() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    distribution = Distribution(int_distribution=Distribution.IntDistribution(low=1, high=3))
    expected_value = ParameterValue(int_value=2)
    sampler.sample.return_value = expected_value
    trial = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    value = trial.suggest_parameter(name="foo", distribution=distribution)
    assert value == expected_value
    assert sampler.sample.call_args_list == [call(distribution=distribution)]
    assert storage.mock_calls == []


def test_suggest_parameter_skip_sample_when_already_suggested() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    distribution = Distribution(int_distribution=Distribution.IntDistribution(low=1, high=3))
    expected_value = ParameterValue(int_value=2)
    sampler.sample.return_value = expected_value
    trial = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    value = trial.suggest_parameter(name="foo", distribution=distribution)
    assert value == expected_value
    assert sampler.sample.call_args_list == [call(distribution=distribution)]
    sampler.reset_mock()
    value = trial.suggest_parameter(name="foo", distribution=distribution)
    assert value == expected_value
    assert sampler.mock_calls == []
    assert storage.mock_calls == []


def test_suggest_parameter_skip_sample_when_fixed() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    distribution = Distribution(int_distribution=Distribution.IntDistribution(low=1, high=3))
    expected_value = ParameterValue(int_value=2)
    unexpected_value = ParameterValue(int_value=3)
    sampler.sample.return_value = unexpected_value
    trial = Trial(
        trial_proto=TrialProto(parameters={"foo": Parameter(value=expected_value)}),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    value = trial.suggest_parameter(name="foo", distribution=distribution)
    assert value == expected_value
    assert sampler.mock_calls == []
    assert storage.mock_calls == []


def test_reset_method_resets_suggested_parameters() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    distribution = Distribution(int_distribution=Distribution.IntDistribution(low=1, high=3))
    original_value = ParameterValue(int_value=2)
    sampler.sample.return_value = original_value
    trial = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    trial.suggest_parameter(name="foo", distribution=distribution)
    trial.reset(hard=False, reload=False)
    new_value = ParameterValue(int_value=3)
    sampler.sample.return_value = new_value
    value = trial.suggest_parameter(name="foo", distribution=distribution)
    assert value == new_value


def test_reset_method_does_not_reset_fixed_parameters() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    sampler.joint_sample.return_value = {}
    distribution = Distribution(int_distribution=Distribution.IntDistribution(low=1, high=3))
    original_value = ParameterValue(int_value=2)
    trial = Trial(
        trial_proto=TrialProto(parameters={"foo": Parameter(value=original_value)}),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    trial.reset(hard=False, reload=False)
    new_value = ParameterValue(int_value=3)
    sampler.sample.return_value = new_value
    value = trial.suggest_parameter(name="foo", distribution=distribution)
    assert value == original_value


def test_hard_reset_does_reset_fixed_parameters() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    distribution = Distribution(int_distribution=Distribution.IntDistribution(low=1, high=3))
    original_value = ParameterValue(int_value=2)
    trial = Trial(
        trial_proto=TrialProto(parameters={"foo": Parameter(value=original_value)}),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    trial.reset(hard=True, reload=False)
    new_value = ParameterValue(int_value=3)
    sampler.sample.return_value = new_value
    value = trial.suggest_parameter(name="foo", distribution=distribution)
    assert value == new_value


def test_reset_method_calls_joint_sample() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    trial = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    trial.reset(hard=False, reload=False)
    # TODO(tsuzuku): Test arguments.
    assert len(sampler.joint_sample.call_args_list) > 0


def test_reset_does_not_sync_sampler() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    trial = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    trial.reset(hard=True, reload=False)
    assert not storage.mock_calls
    assert len(sampler.sync.call_args_list) == 0


def test_hard_reset_does_sync_sampler() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    trial = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    trial.reset(hard=True, reload=False)
    assert not storage.mock_calls
    assert len(sampler.sync.call_args_list) == 0


def test_reset_with_reload_sync_sampler() -> None:
    sampler = MagicMock()
    storage = MagicMock()
    trial = Trial(
        trial_proto=TrialProto(),
        study_info=StudyInfo(),
        storage=storage,
        sampler=sampler,
    )
    trial.reset(hard=False, reload=True)
    assert len(storage.get_trials.call_args_list) > 0
    assert len(sampler.sync.call_args_list) > 0

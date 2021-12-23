# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from typing import Iterable as typing___Iterable
from typing import Mapping as typing___Mapping
from typing import MutableMapping as typing___MutableMapping
from typing import NewType as typing___NewType
from typing import Optional as typing___Optional
from typing import Text as typing___Text
from typing import cast as typing___cast

from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)
from google.protobuf.descriptor import (
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
)
from google.protobuf.descriptor import (
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
)
from google.protobuf.internal.enum_type_wrapper import (
    _EnumTypeWrapper as google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper,
)
from google.protobuf.message import Message as google___protobuf___message___Message
from google.protobuf.timestamp_pb2 import (
    Timestamp as google___protobuf___timestamp_pb2___Timestamp,
)
from typing_extensions import Literal as typing_extensions___Literal

builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int

DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

class StudyInfo(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class UserAttrsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        value: typing___Text = ...
        def __init__(
            self,
            *,
            key: typing___Optional[typing___Text] = None,
            value: typing___Optional[typing___Text] = None,
        ) -> None: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["key", b"key", "value", b"value"]
        ) -> None: ...
    type___UserAttrsEntry = UserAttrsEntry
    class SystemAttrsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        value: typing___Text = ...
        def __init__(
            self,
            *,
            key: typing___Optional[typing___Text] = None,
            value: typing___Optional[typing___Text] = None,
        ) -> None: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["key", b"key", "value", b"value"]
        ) -> None: ...
    type___SystemAttrsEntry = SystemAttrsEntry

    study_id: typing___Text = ...
    study_name: typing___Text = ...
    @property
    def targets(
        self,
    ) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[
        type___Target
    ]: ...
    @property
    def user_attrs(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...
    @property
    def system_attrs(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...
    @property
    def create_time(self) -> google___protobuf___timestamp_pb2___Timestamp: ...
    @property
    def last_update_time(self) -> google___protobuf___timestamp_pb2___Timestamp: ...
    def __init__(
        self,
        *,
        study_id: typing___Optional[typing___Text] = None,
        study_name: typing___Optional[typing___Text] = None,
        targets: typing___Optional[typing___Iterable[type___Target]] = None,
        user_attrs: typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        system_attrs: typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        create_time: typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
        last_update_time: typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing_extensions___Literal[
            "create_time", b"create_time", "last_update_time", b"last_update_time"
        ],
    ) -> builtin___bool: ...
    def ClearField(
        self,
        field_name: typing_extensions___Literal[
            "create_time",
            b"create_time",
            "last_update_time",
            b"last_update_time",
            "study_id",
            b"study_id",
            "study_name",
            b"study_name",
            "system_attrs",
            b"system_attrs",
            "targets",
            b"targets",
            "user_attrs",
            b"user_attrs",
        ],
    ) -> None: ...

type___StudyInfo = StudyInfo

class Target(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    DirectionValue = typing___NewType("DirectionValue", builtin___int)
    type___DirectionValue = DirectionValue
    Direction: _Direction
    class _Direction(
        google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[Target.DirectionValue]
    ):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        UNKNOWN = typing___cast(Target.DirectionValue, 0)
        MAXIMIZE = typing___cast(Target.DirectionValue, 1)
        MINIMIZE = typing___cast(Target.DirectionValue, 2)
    UNKNOWN = typing___cast(Target.DirectionValue, 0)
    MAXIMIZE = typing___cast(Target.DirectionValue, 1)
    MINIMIZE = typing___cast(Target.DirectionValue, 2)
    type___Direction = Direction

    name: typing___Text = ...
    direction: type___Target.DirectionValue = ...
    def __init__(
        self,
        *,
        name: typing___Optional[typing___Text] = None,
        direction: typing___Optional[type___Target.DirectionValue] = None,
    ) -> None: ...
    def ClearField(
        self, field_name: typing_extensions___Literal["direction", b"direction", "name", b"name"]
    ) -> None: ...

type___Target = Target

class Trial(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    StateValue = typing___NewType("StateValue", builtin___int)
    type___StateValue = StateValue
    State: _State
    class _State(
        google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[Trial.StateValue]
    ):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        UNKNOWN = typing___cast(Trial.StateValue, 0)
        CREATED = typing___cast(Trial.StateValue, 1)
        WAITING = typing___cast(Trial.StateValue, 2)
        RUNNING = typing___cast(Trial.StateValue, 3)
        COMPLETED = typing___cast(Trial.StateValue, 4)
        FAILED = typing___cast(Trial.StateValue, 5)
        PRUNED = typing___cast(Trial.StateValue, 6)
        PARTIALLY_COMPLETED = typing___cast(Trial.StateValue, 7)
        PARTIALLY_FAILED = typing___cast(Trial.StateValue, 8)
    UNKNOWN = typing___cast(Trial.StateValue, 0)
    CREATED = typing___cast(Trial.StateValue, 1)
    WAITING = typing___cast(Trial.StateValue, 2)
    RUNNING = typing___cast(Trial.StateValue, 3)
    COMPLETED = typing___cast(Trial.StateValue, 4)
    FAILED = typing___cast(Trial.StateValue, 5)
    PRUNED = typing___cast(Trial.StateValue, 6)
    PARTIALLY_COMPLETED = typing___cast(Trial.StateValue, 7)
    PARTIALLY_FAILED = typing___cast(Trial.StateValue, 8)
    type___State = State
    class UserAttrsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        value: typing___Text = ...
        def __init__(
            self,
            *,
            key: typing___Optional[typing___Text] = None,
            value: typing___Optional[typing___Text] = None,
        ) -> None: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["key", b"key", "value", b"value"]
        ) -> None: ...
    type___UserAttrsEntry = UserAttrsEntry
    class SystemAttrsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        value: typing___Text = ...
        def __init__(
            self,
            *,
            key: typing___Optional[typing___Text] = None,
            value: typing___Optional[typing___Text] = None,
        ) -> None: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["key", b"key", "value", b"value"]
        ) -> None: ...
    type___SystemAttrsEntry = SystemAttrsEntry
    class ParametersEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        @property
        def value(self) -> type___Parameter: ...
        def __init__(
            self,
            *,
            key: typing___Optional[typing___Text] = None,
            value: typing___Optional[type___Parameter] = None,
        ) -> None: ...
        def HasField(
            self, field_name: typing_extensions___Literal["value", b"value"]
        ) -> builtin___bool: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["key", b"key", "value", b"value"]
        ) -> None: ...
    type___ParametersEntry = ParametersEntry
    class DistributionsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        @property
        def value(self) -> type___Distribution: ...
        def __init__(
            self,
            *,
            key: typing___Optional[typing___Text] = None,
            value: typing___Optional[type___Distribution] = None,
        ) -> None: ...
        def HasField(
            self, field_name: typing_extensions___Literal["value", b"value"]
        ) -> builtin___bool: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["key", b"key", "value", b"value"]
        ) -> None: ...
    type___DistributionsEntry = DistributionsEntry

    trial_id: typing___Text = ...
    study_id: typing___Text = ...
    last_known_state: type___Trial.StateValue = ...
    @property
    def user_attrs(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...
    @property
    def system_attrs(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...
    @property
    def parameters(self) -> typing___MutableMapping[typing___Text, type___Parameter]: ...
    @property
    def values(
        self,
    ) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[
        type___ObjectiveValue
    ]: ...
    @property
    def reports(
        self,
    ) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[
        type___Report
    ]: ...
    @property
    def worker_id(self) -> type___WorkerID: ...
    @property
    def create_time(self) -> google___protobuf___timestamp_pb2___Timestamp: ...
    @property
    def last_update_time(self) -> google___protobuf___timestamp_pb2___Timestamp: ...
    @property
    def distributions(self) -> typing___MutableMapping[typing___Text, type___Distribution]: ...
    def __init__(
        self,
        *,
        trial_id: typing___Optional[typing___Text] = None,
        study_id: typing___Optional[typing___Text] = None,
        last_known_state: typing___Optional[type___Trial.StateValue] = None,
        user_attrs: typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        system_attrs: typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        parameters: typing___Optional[typing___Mapping[typing___Text, type___Parameter]] = None,
        values: typing___Optional[typing___Iterable[type___ObjectiveValue]] = None,
        reports: typing___Optional[typing___Iterable[type___Report]] = None,
        worker_id: typing___Optional[type___WorkerID] = None,
        create_time: typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
        last_update_time: typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
        distributions: typing___Optional[
            typing___Mapping[typing___Text, type___Distribution]
        ] = None,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing_extensions___Literal[
            "create_time",
            b"create_time",
            "last_update_time",
            b"last_update_time",
            "worker_id",
            b"worker_id",
        ],
    ) -> builtin___bool: ...
    def ClearField(
        self,
        field_name: typing_extensions___Literal[
            "create_time",
            b"create_time",
            "distributions",
            b"distributions",
            "last_known_state",
            b"last_known_state",
            "last_update_time",
            b"last_update_time",
            "parameters",
            b"parameters",
            "reports",
            b"reports",
            "study_id",
            b"study_id",
            "system_attrs",
            b"system_attrs",
            "trial_id",
            b"trial_id",
            "user_attrs",
            b"user_attrs",
            "values",
            b"values",
            "worker_id",
            b"worker_id",
        ],
    ) -> None: ...

type___Trial = Trial

class WorkerID(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    client_id: typing___Text = ...
    thread_id: builtin___int = ...
    bucket_id: builtin___int = ...
    def __init__(
        self,
        *,
        client_id: typing___Optional[typing___Text] = None,
        thread_id: typing___Optional[builtin___int] = None,
        bucket_id: typing___Optional[builtin___int] = None,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing_extensions___Literal[
            "bucket_id", b"bucket_id", "client_id", b"client_id", "thread_id", b"thread_id"
        ],
    ) -> None: ...

type___WorkerID = WorkerID

class Report(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    step: builtin___int = ...
    @property
    def event_time(self) -> google___protobuf___timestamp_pb2___Timestamp: ...
    @property
    def values(
        self,
    ) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[
        type___ObjectiveValue
    ]: ...
    def __init__(
        self,
        *,
        event_time: typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
        step: typing___Optional[builtin___int] = None,
        values: typing___Optional[typing___Iterable[type___ObjectiveValue]] = None,
    ) -> None: ...
    def HasField(
        self, field_name: typing_extensions___Literal["event_time", b"event_time"]
    ) -> builtin___bool: ...
    def ClearField(
        self,
        field_name: typing_extensions___Literal[
            "event_time", b"event_time", "step", b"step", "values", b"values"
        ],
    ) -> None: ...

type___Report = Report

class ObjectiveValue(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    StatusValue = typing___NewType("StatusValue", builtin___int)
    type___StatusValue = StatusValue
    Status: _Status
    class _Status(
        google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[
            ObjectiveValue.StatusValue
        ]
    ):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        UNKNOWN = typing___cast(ObjectiveValue.StatusValue, 0)
        VALID = typing___cast(ObjectiveValue.StatusValue, 1)
        SKIPPED = typing___cast(ObjectiveValue.StatusValue, 2)
        INFEASIBLE = typing___cast(ObjectiveValue.StatusValue, 3)
        NAN = typing___cast(ObjectiveValue.StatusValue, 4)
        INF = typing___cast(ObjectiveValue.StatusValue, 5)
        NEGATIVE_INF = typing___cast(ObjectiveValue.StatusValue, 6)
    UNKNOWN = typing___cast(ObjectiveValue.StatusValue, 0)
    VALID = typing___cast(ObjectiveValue.StatusValue, 1)
    SKIPPED = typing___cast(ObjectiveValue.StatusValue, 2)
    INFEASIBLE = typing___cast(ObjectiveValue.StatusValue, 3)
    NAN = typing___cast(ObjectiveValue.StatusValue, 4)
    INF = typing___cast(ObjectiveValue.StatusValue, 5)
    NEGATIVE_INF = typing___cast(ObjectiveValue.StatusValue, 6)
    type___Status = Status

    value: builtin___float = ...
    status: type___ObjectiveValue.StatusValue = ...
    def __init__(
        self,
        *,
        value: typing___Optional[builtin___float] = None,
        status: typing___Optional[type___ObjectiveValue.StatusValue] = None,
    ) -> None: ...
    def ClearField(
        self, field_name: typing_extensions___Literal["status", b"status", "value", b"value"]
    ) -> None: ...

type___ObjectiveValue = ObjectiveValue

class Parameter(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    @property
    def value(self) -> type___ParameterValue: ...
    def __init__(
        self,
        *,
        value: typing___Optional[type___ParameterValue] = None,
    ) -> None: ...
    def HasField(
        self, field_name: typing_extensions___Literal["value", b"value"]
    ) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal["value", b"value"]) -> None: ...

type___Parameter = Parameter

class ParameterValue(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    int_value: builtin___int = ...
    double_value: builtin___float = ...
    string_value: typing___Text = ...
    def __init__(
        self,
        *,
        int_value: typing___Optional[builtin___int] = None,
        double_value: typing___Optional[builtin___float] = None,
        string_value: typing___Optional[typing___Text] = None,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing_extensions___Literal[
            "double_value",
            b"double_value",
            "int_value",
            b"int_value",
            "string_value",
            b"string_value",
            "value",
            b"value",
        ],
    ) -> builtin___bool: ...
    def ClearField(
        self,
        field_name: typing_extensions___Literal[
            "double_value",
            b"double_value",
            "int_value",
            b"int_value",
            "string_value",
            b"string_value",
            "value",
            b"value",
        ],
    ) -> None: ...
    def WhichOneof(
        self, oneof_group: typing_extensions___Literal["value", b"value"]
    ) -> typing_extensions___Literal["int_value", "double_value", "string_value"]: ...

type___ParameterValue = ParameterValue

class Distribution(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class FloatDistribution(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        low: builtin___float = ...
        high: builtin___float = ...
        log_scale: builtin___bool = ...
        def __init__(
            self,
            *,
            low: typing___Optional[builtin___float] = None,
            high: typing___Optional[builtin___float] = None,
            log_scale: typing___Optional[builtin___bool] = None,
        ) -> None: ...
        def ClearField(
            self,
            field_name: typing_extensions___Literal[
                "high", b"high", "log_scale", b"log_scale", "low", b"low"
            ],
        ) -> None: ...
    type___FloatDistribution = FloatDistribution
    class IntDistribution(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        low: builtin___int = ...
        high: builtin___int = ...
        log_scale: builtin___bool = ...
        def __init__(
            self,
            *,
            low: typing___Optional[builtin___int] = None,
            high: typing___Optional[builtin___int] = None,
            log_scale: typing___Optional[builtin___bool] = None,
        ) -> None: ...
        def ClearField(
            self,
            field_name: typing_extensions___Literal[
                "high", b"high", "log_scale", b"log_scale", "low", b"low"
            ],
        ) -> None: ...
    type___IntDistribution = IntDistribution
    class CategoricalDistribution(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        @property
        def choices(
            self,
        ) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[
            type___ParameterValue
        ]: ...
        def __init__(
            self,
            *,
            choices: typing___Optional[typing___Iterable[type___ParameterValue]] = None,
        ) -> None: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["choices", b"choices"]
        ) -> None: ...
    type___CategoricalDistribution = CategoricalDistribution
    @property
    def float_distribution(self) -> type___Distribution.FloatDistribution: ...
    @property
    def int_distribution(self) -> type___Distribution.IntDistribution: ...
    @property
    def categorical_distribution(self) -> type___Distribution.CategoricalDistribution: ...
    def __init__(
        self,
        *,
        float_distribution: typing___Optional[type___Distribution.FloatDistribution] = None,
        int_distribution: typing___Optional[type___Distribution.IntDistribution] = None,
        categorical_distribution: typing___Optional[
            type___Distribution.CategoricalDistribution
        ] = None,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing_extensions___Literal[
            "categorical_distribution",
            b"categorical_distribution",
            "distribution",
            b"distribution",
            "float_distribution",
            b"float_distribution",
            "int_distribution",
            b"int_distribution",
        ],
    ) -> builtin___bool: ...
    def ClearField(
        self,
        field_name: typing_extensions___Literal[
            "categorical_distribution",
            b"categorical_distribution",
            "distribution",
            b"distribution",
            "float_distribution",
            b"float_distribution",
            "int_distribution",
            b"int_distribution",
        ],
    ) -> None: ...
    def WhichOneof(
        self, oneof_group: typing_extensions___Literal["distribution", b"distribution"]
    ) -> typing_extensions___Literal[
        "float_distribution", "int_distribution", "categorical_distribution"
    ]: ...

type___Distribution = Distribution

class SearchSpace(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class DistributionsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        @property
        def value(self) -> type___Distribution: ...
        def __init__(
            self,
            *,
            key: typing___Optional[typing___Text] = None,
            value: typing___Optional[type___Distribution] = None,
        ) -> None: ...
        def HasField(
            self, field_name: typing_extensions___Literal["value", b"value"]
        ) -> builtin___bool: ...
        def ClearField(
            self, field_name: typing_extensions___Literal["key", b"key", "value", b"value"]
        ) -> None: ...
    type___DistributionsEntry = DistributionsEntry
    @property
    def distributions(self) -> typing___MutableMapping[typing___Text, type___Distribution]: ...
    def __init__(
        self,
        *,
        distributions: typing___Optional[
            typing___Mapping[typing___Text, type___Distribution]
        ] = None,
    ) -> None: ...
    def ClearField(
        self, field_name: typing_extensions___Literal["distributions", b"distributions"]
    ) -> None: ...

type___SearchSpace = SearchSpace
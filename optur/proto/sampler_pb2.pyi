# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Optional as typing___Optional,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

class SamplerConfig(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def random(self) -> type___RandomSamplerConfig: ...

    def __init__(self,
        *,
        random : typing___Optional[type___RandomSamplerConfig] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"random",b"random",u"sampler",b"sampler"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"random",b"random",u"sampler",b"sampler"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions___Literal[u"sampler",b"sampler"]) -> typing_extensions___Literal["random"]: ...
type___SamplerConfig = SamplerConfig

class RandomSamplerConfig(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    def __init__(self,
        ) -> None: ...
type___RandomSamplerConfig = RandomSamplerConfig

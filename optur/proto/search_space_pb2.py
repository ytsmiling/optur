# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: optur/proto/search_space.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="optur/proto/search_space.proto",
    package="optur",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x1eoptur/proto/search_space.proto\x12\x05optur"^\n\x0eParameterValue\x12\x13\n\tint_value\x18\x01 \x01(\x03H\x00\x12\x16\n\x0c\x64ouble_value\x18\x02 \x01(\x01H\x00\x12\x16\n\x0cstring_value\x18\x03 \x01(\tH\x00\x42\x07\n\x05value"\xbd\x04\n\x0c\x44istribution\x12\x43\n\x12\x66loat_distribution\x18\x01 \x01(\x0b\x32%.optur.Distribution.FloatDistributionH\x00\x12?\n\x10int_distribution\x18\x02 \x01(\x0b\x32#.optur.Distribution.IntDistributionH\x00\x12O\n\x18\x63\x61tegorical_distribution\x18\x03 \x01(\x0b\x32+.optur.Distribution.CategoricalDistributionH\x00\x12\x43\n\x12\x66ixed_distribution\x18\x04 \x01(\x0b\x32%.optur.Distribution.FixedDistributionH\x00\x1a\x41\n\x11\x46loatDistribution\x12\x0b\n\x03low\x18\x01 \x01(\x02\x12\x0c\n\x04high\x18\x02 \x01(\x02\x12\x11\n\tlog_scale\x18\x03 \x01(\x08\x1a?\n\x0fIntDistribution\x12\x0b\n\x03low\x18\x01 \x01(\x03\x12\x0c\n\x04high\x18\x02 \x01(\x03\x12\x11\n\tlog_scale\x18\x03 \x01(\x08\x1a\x41\n\x17\x43\x61tegoricalDistribution\x12&\n\x07\x63hoices\x18\x01 \x03(\x0b\x32\x15.optur.ParameterValue\x1a:\n\x11\x46ixedDistribution\x12%\n\x06values\x18\x01 \x03(\x0b\x32\x15.optur.ParameterValueB\x0e\n\x0c\x64istribution"\x96\x01\n\x0bSearchSpace\x12<\n\rdistributions\x18\x01 \x03(\x0b\x32%.optur.SearchSpace.DistributionsEntry\x1aI\n\x12\x44istributionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12"\n\x05value\x18\x02 \x01(\x0b\x32\x13.optur.Distribution:\x02\x38\x01\x62\x06proto3',
)


_PARAMETERVALUE = _descriptor.Descriptor(
    name="ParameterValue",
    full_name="optur.ParameterValue",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="int_value",
            full_name="optur.ParameterValue.int_value",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="double_value",
            full_name="optur.ParameterValue.double_value",
            index=1,
            number=2,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="string_value",
            full_name="optur.ParameterValue.string_value",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name="value",
            full_name="optur.ParameterValue.value",
            index=0,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
    ],
    serialized_start=41,
    serialized_end=135,
)


_DISTRIBUTION_FLOATDISTRIBUTION = _descriptor.Descriptor(
    name="FloatDistribution",
    full_name="optur.Distribution.FloatDistribution",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="low",
            full_name="optur.Distribution.FloatDistribution.low",
            index=0,
            number=1,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="high",
            full_name="optur.Distribution.FloatDistribution.high",
            index=1,
            number=2,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="log_scale",
            full_name="optur.Distribution.FloatDistribution.log_scale",
            index=2,
            number=3,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=438,
    serialized_end=503,
)

_DISTRIBUTION_INTDISTRIBUTION = _descriptor.Descriptor(
    name="IntDistribution",
    full_name="optur.Distribution.IntDistribution",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="low",
            full_name="optur.Distribution.IntDistribution.low",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="high",
            full_name="optur.Distribution.IntDistribution.high",
            index=1,
            number=2,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="log_scale",
            full_name="optur.Distribution.IntDistribution.log_scale",
            index=2,
            number=3,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=505,
    serialized_end=568,
)

_DISTRIBUTION_CATEGORICALDISTRIBUTION = _descriptor.Descriptor(
    name="CategoricalDistribution",
    full_name="optur.Distribution.CategoricalDistribution",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="choices",
            full_name="optur.Distribution.CategoricalDistribution.choices",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=570,
    serialized_end=635,
)

_DISTRIBUTION_FIXEDDISTRIBUTION = _descriptor.Descriptor(
    name="FixedDistribution",
    full_name="optur.Distribution.FixedDistribution",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="values",
            full_name="optur.Distribution.FixedDistribution.values",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=637,
    serialized_end=695,
)

_DISTRIBUTION = _descriptor.Descriptor(
    name="Distribution",
    full_name="optur.Distribution",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="float_distribution",
            full_name="optur.Distribution.float_distribution",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="int_distribution",
            full_name="optur.Distribution.int_distribution",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="categorical_distribution",
            full_name="optur.Distribution.categorical_distribution",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="fixed_distribution",
            full_name="optur.Distribution.fixed_distribution",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _DISTRIBUTION_FLOATDISTRIBUTION,
        _DISTRIBUTION_INTDISTRIBUTION,
        _DISTRIBUTION_CATEGORICALDISTRIBUTION,
        _DISTRIBUTION_FIXEDDISTRIBUTION,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name="distribution",
            full_name="optur.Distribution.distribution",
            index=0,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
    ],
    serialized_start=138,
    serialized_end=711,
)


_SEARCHSPACE_DISTRIBUTIONSENTRY = _descriptor.Descriptor(
    name="DistributionsEntry",
    full_name="optur.SearchSpace.DistributionsEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="optur.SearchSpace.DistributionsEntry.key",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="value",
            full_name="optur.SearchSpace.DistributionsEntry.value",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=b"8\001",
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=791,
    serialized_end=864,
)

_SEARCHSPACE = _descriptor.Descriptor(
    name="SearchSpace",
    full_name="optur.SearchSpace",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="distributions",
            full_name="optur.SearchSpace.distributions",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _SEARCHSPACE_DISTRIBUTIONSENTRY,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=714,
    serialized_end=864,
)

_PARAMETERVALUE.oneofs_by_name["value"].fields.append(_PARAMETERVALUE.fields_by_name["int_value"])
_PARAMETERVALUE.fields_by_name["int_value"].containing_oneof = _PARAMETERVALUE.oneofs_by_name[
    "value"
]
_PARAMETERVALUE.oneofs_by_name["value"].fields.append(
    _PARAMETERVALUE.fields_by_name["double_value"]
)
_PARAMETERVALUE.fields_by_name["double_value"].containing_oneof = _PARAMETERVALUE.oneofs_by_name[
    "value"
]
_PARAMETERVALUE.oneofs_by_name["value"].fields.append(
    _PARAMETERVALUE.fields_by_name["string_value"]
)
_PARAMETERVALUE.fields_by_name["string_value"].containing_oneof = _PARAMETERVALUE.oneofs_by_name[
    "value"
]
_DISTRIBUTION_FLOATDISTRIBUTION.containing_type = _DISTRIBUTION
_DISTRIBUTION_INTDISTRIBUTION.containing_type = _DISTRIBUTION
_DISTRIBUTION_CATEGORICALDISTRIBUTION.fields_by_name["choices"].message_type = _PARAMETERVALUE
_DISTRIBUTION_CATEGORICALDISTRIBUTION.containing_type = _DISTRIBUTION
_DISTRIBUTION_FIXEDDISTRIBUTION.fields_by_name["values"].message_type = _PARAMETERVALUE
_DISTRIBUTION_FIXEDDISTRIBUTION.containing_type = _DISTRIBUTION
_DISTRIBUTION.fields_by_name["float_distribution"].message_type = _DISTRIBUTION_FLOATDISTRIBUTION
_DISTRIBUTION.fields_by_name["int_distribution"].message_type = _DISTRIBUTION_INTDISTRIBUTION
_DISTRIBUTION.fields_by_name[
    "categorical_distribution"
].message_type = _DISTRIBUTION_CATEGORICALDISTRIBUTION
_DISTRIBUTION.fields_by_name["fixed_distribution"].message_type = _DISTRIBUTION_FIXEDDISTRIBUTION
_DISTRIBUTION.oneofs_by_name["distribution"].fields.append(
    _DISTRIBUTION.fields_by_name["float_distribution"]
)
_DISTRIBUTION.fields_by_name["float_distribution"].containing_oneof = _DISTRIBUTION.oneofs_by_name[
    "distribution"
]
_DISTRIBUTION.oneofs_by_name["distribution"].fields.append(
    _DISTRIBUTION.fields_by_name["int_distribution"]
)
_DISTRIBUTION.fields_by_name["int_distribution"].containing_oneof = _DISTRIBUTION.oneofs_by_name[
    "distribution"
]
_DISTRIBUTION.oneofs_by_name["distribution"].fields.append(
    _DISTRIBUTION.fields_by_name["categorical_distribution"]
)
_DISTRIBUTION.fields_by_name[
    "categorical_distribution"
].containing_oneof = _DISTRIBUTION.oneofs_by_name["distribution"]
_DISTRIBUTION.oneofs_by_name["distribution"].fields.append(
    _DISTRIBUTION.fields_by_name["fixed_distribution"]
)
_DISTRIBUTION.fields_by_name["fixed_distribution"].containing_oneof = _DISTRIBUTION.oneofs_by_name[
    "distribution"
]
_SEARCHSPACE_DISTRIBUTIONSENTRY.fields_by_name["value"].message_type = _DISTRIBUTION
_SEARCHSPACE_DISTRIBUTIONSENTRY.containing_type = _SEARCHSPACE
_SEARCHSPACE.fields_by_name["distributions"].message_type = _SEARCHSPACE_DISTRIBUTIONSENTRY
DESCRIPTOR.message_types_by_name["ParameterValue"] = _PARAMETERVALUE
DESCRIPTOR.message_types_by_name["Distribution"] = _DISTRIBUTION
DESCRIPTOR.message_types_by_name["SearchSpace"] = _SEARCHSPACE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ParameterValue = _reflection.GeneratedProtocolMessageType(
    "ParameterValue",
    (_message.Message,),
    {
        "DESCRIPTOR": _PARAMETERVALUE,
        "__module__": "optur.proto.search_space_pb2"
        # @@protoc_insertion_point(class_scope:optur.ParameterValue)
    },
)
_sym_db.RegisterMessage(ParameterValue)

Distribution = _reflection.GeneratedProtocolMessageType(
    "Distribution",
    (_message.Message,),
    {
        "FloatDistribution": _reflection.GeneratedProtocolMessageType(
            "FloatDistribution",
            (_message.Message,),
            {
                "DESCRIPTOR": _DISTRIBUTION_FLOATDISTRIBUTION,
                "__module__": "optur.proto.search_space_pb2"
                # @@protoc_insertion_point(class_scope:optur.Distribution.FloatDistribution)
            },
        ),
        "IntDistribution": _reflection.GeneratedProtocolMessageType(
            "IntDistribution",
            (_message.Message,),
            {
                "DESCRIPTOR": _DISTRIBUTION_INTDISTRIBUTION,
                "__module__": "optur.proto.search_space_pb2"
                # @@protoc_insertion_point(class_scope:optur.Distribution.IntDistribution)
            },
        ),
        "CategoricalDistribution": _reflection.GeneratedProtocolMessageType(
            "CategoricalDistribution",
            (_message.Message,),
            {
                "DESCRIPTOR": _DISTRIBUTION_CATEGORICALDISTRIBUTION,
                "__module__": "optur.proto.search_space_pb2"
                # @@protoc_insertion_point(class_scope:optur.Distribution.CategoricalDistribution)
            },
        ),
        "FixedDistribution": _reflection.GeneratedProtocolMessageType(
            "FixedDistribution",
            (_message.Message,),
            {
                "DESCRIPTOR": _DISTRIBUTION_FIXEDDISTRIBUTION,
                "__module__": "optur.proto.search_space_pb2"
                # @@protoc_insertion_point(class_scope:optur.Distribution.FixedDistribution)
            },
        ),
        "DESCRIPTOR": _DISTRIBUTION,
        "__module__": "optur.proto.search_space_pb2"
        # @@protoc_insertion_point(class_scope:optur.Distribution)
    },
)
_sym_db.RegisterMessage(Distribution)
_sym_db.RegisterMessage(Distribution.FloatDistribution)
_sym_db.RegisterMessage(Distribution.IntDistribution)
_sym_db.RegisterMessage(Distribution.CategoricalDistribution)
_sym_db.RegisterMessage(Distribution.FixedDistribution)

SearchSpace = _reflection.GeneratedProtocolMessageType(
    "SearchSpace",
    (_message.Message,),
    {
        "DistributionsEntry": _reflection.GeneratedProtocolMessageType(
            "DistributionsEntry",
            (_message.Message,),
            {
                "DESCRIPTOR": _SEARCHSPACE_DISTRIBUTIONSENTRY,
                "__module__": "optur.proto.search_space_pb2"
                # @@protoc_insertion_point(class_scope:optur.SearchSpace.DistributionsEntry)
            },
        ),
        "DESCRIPTOR": _SEARCHSPACE,
        "__module__": "optur.proto.search_space_pb2"
        # @@protoc_insertion_point(class_scope:optur.SearchSpace)
    },
)
_sym_db.RegisterMessage(SearchSpace)
_sym_db.RegisterMessage(SearchSpace.DistributionsEntry)


_SEARCHSPACE_DISTRIBUTIONSENTRY._options = None
# @@protoc_insertion_point(module_scope)

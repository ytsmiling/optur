from typing import Dict

from optur.proto.study_pb2 import ParameterValue
from optur.proto.study_pb2 import Trial as TrialProto


class Trial:
    def __init__(self, trial_proto: TrialProto) -> None:
        pass

    def update_parameters(self, parameters: Dict[str, ParameterValue]) -> None:
        pass

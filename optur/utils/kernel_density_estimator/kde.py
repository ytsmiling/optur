import abc
from typing import Dict, List, Optional, Sequence

from optur.proto.search_space_pb2 import Observation, ParameterValue, SearchSpace


class KDE(abc.ABC):
    @abc.abstractclassmethod
    def init(self, search_space: SearchSpace, observations: Sequence[Observation]) -> None:
        pass

    @abc.abstractclassmethod
    def log_pdf(self, sample: Observation) -> float:
        pass

    @abc.abstractclassmethod
    def sample(
        self, k: int, fixed_parameters: Optional[Dict[str, ParameterValue]]
    ) -> List[Observation]:
        pass

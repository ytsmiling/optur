import abc
from typing import List, Sequence

from optur.proto.search_space_pb2 import Observation, SearchSpace


class KDE(abc.ABC):
    @abc.abstractclassmethod
    def init(self, search_space: SearchSpace, observations: Sequence[Observation]) -> None:
        pass

    @abc.abstractclassmethod
    def log_pdf(self, sample: Observation) -> float:
        pass

    @abc.abstractclassmethod
    def sample(self, k: int) -> List[Observation]:
        pass

from typing import List, Sequence

from optur.proto.search_space_pb2 import Observation, SearchSpace
from optur.utils.kernel_density_estimator.kde import KDE


# The Gaussian kernel is used for continuous parameters.
# The Aitchison-Aitken kernel is used for categorical parameters.
class FactorizedKDE(KDE):
    def init(self, search_space: SearchSpace, observations: Sequence[Observation]) -> None:
        pass

    def log_pdf(self, sample: Observation) -> float:
        pass

    def sample(self, k: int) -> List[Observation]:
        pass

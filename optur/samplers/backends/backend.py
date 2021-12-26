import abc
from typing import Dict, Optional, Sequence

from optur.proto.study_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import Trial as TrialProto


class SamplerBackend(abc.ABC):
    def sync(self, trials: Sequence[TrialProto]) -> None:
        """Update sampler-specific cache with the trials.

        Backends can ignore trials that have already be synced with some finished states.
        """
        pass

    def joint_sample(
        self,
        fixed_parameters: Optional[Dict[str, ParameterValue]],
        search_space: Optional[SearchSpace],
    ) -> Dict[str, ParameterValue]:
        """Perform joint-sampling for the trial.

        When ``fixed_parameters`` is not :obj:`None`, the return value must includes them.

        When ``search_space`` is not :obj:`None`, this method uses the ``search_space``
        for the joint-sampling.
        When ``search_space`` is :obj:`None`, this method infers the search space from
        past trials. This process is called `intersection_searchspace` in optuna, but
        optur does not necessarily infer the same search space with optuna.

        This method might use an internal cache, but this method never updates internal
        states including the cache.
        """

        # Default implementation for samplers that cannot take inter-parameter
        # correlations in to account.
        ret = fixed_parameters.copy() if fixed_parameters is not None else {}
        if search_space:
            for key, distribution in search_space.distributions.items():
                if not fixed_parameters or key not in fixed_parameters:
                    ret[key] = self.sample(distribution=distribution)
        return ret

    @abc.abstractclassmethod
    def sample(self, distribution: Distribution) -> ParameterValue:
        """Sample a parameter.

        This method might use an internal cache, but this method never updates internal
        states including the cache.
        """
        pass

import abc
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import numpy.typing as npt
except ImportError:
    pass

from optur.proto.sampler_pb2 import RandomSamplerConfig, SamplerConfig
from optur.proto.search_space_pb2 import Distribution, ParameterValue, SearchSpace
from optur.proto.study_pb2 import AttributeValue
from optur.proto.study_pb2 import Trial as TrialProto
from optur.samplers.random import RandomSampler
from optur.samplers.sampler import JointSampleResult, Sampler
from optur.utils.search_space_tracker import SearchSpaceTracker
from optur.utils.sorted_trials import SortedTrials

_N_RERFERENCED_TRIALS_KEY = "smpl.tpe.n"


class TPESampler(Sampler):
    def __init__(self, sampler_config: SamplerConfig) -> None:
        super().__init__(sampler_config=sampler_config)
        assert sampler_config.HasField("tpe")
        self._tpe_config = sampler_config.tpe
        self._fallback_sampler = RandomSampler(SamplerConfig(random=RandomSamplerConfig()))
        self._search_space_tracker = SearchSpaceTracker(search_space=None)
        self._sorted_trials: SortedTrials

    def set_search_space(self, search_space: Optional[SearchSpace]) -> None:
        # TODO(tsuzuku): Reset cache (e.g., sorted trials).
        self._search_space_tracker = SearchSpaceTracker(search_space=search_space)
        # We need all past trials in the next sync because we cleared the cache.
        self.update_timestamp(timestamp=None)

    def sync(self, trials: Sequence[TrialProto]) -> None:
        self._fallback_sampler.sync(trials=trials)
        self._sorted_trials.sync(trials=trials)
        self._search_space_tracker.sync(trials=trials)

    def joint_sample(
        self,
        fixed_parameters: Optional[Dict[str, ParameterValue]] = None,
    ) -> JointSampleResult:
        sorted_trials = self._sorted_trials.to_list()
        if len(sorted_trials) < self._tpe_config.n_startup_trials:
            return JointSampleResult(parameters={}, system_attrs={})
        search_space = self._search_space_tracker.current_search_space
        # TODO(tsuzuku): Extend to MOTPE.
        half_idx = len(sorted_trials) // 2
        _less_half_trials = sorted_trials[:half_idx]  # D_l
        _greater_half_trials = sorted_trials[half_idx:]  # D_g
        # TODO(tsuzuku): Calculate weights.
        kde_l = _UnivariateKDE(
            search_space=search_space, trials=_less_half_trials, weights=np.ones(())
        )
        kde_g = _UnivariateKDE(
            search_space=search_space, trials=_greater_half_trials, weights=np.ones(())
        )
        samples = kde_l.sample(
            fixed_parameters=fixed_parameters or {}, k=self._tpe_config.n_ei_candidates
        )
        log_pdf_l = kde_l.log_pdf(samples)
        log_pdf_g = kde_g.log_pdf(samples)
        best_sample_idx = np.argmax(log_pdf_l - log_pdf_g)
        best_sample = {name: sample[name][best_sample_idx] for name, sample in samples.items()}
        return JointSampleResult(
            parameters=kde_l.sample_to_value(best_sample),
            system_attrs={
                _N_RERFERENCED_TRIALS_KEY: AttributeValue(
                    int_value=self._sorted_trials.n_trials()
                ),
            },
        )

    def sample(self, distribution: Distribution) -> ParameterValue:
        return self._fallback_sampler.sample(distribution=distribution)


# The Gaussian kernel is used for continuous parameters.
# The Aitchison-Aitken kernel is used for categorical parameters.
class _UnivariateKDE:
    def __init__(
        self,
        search_space: SearchSpace,
        trials: Sequence[TrialProto],
        weights: "npt.NDArray[np.float64]",
    ) -> None:
        self._search_space = search_space
        n_distribution = len(search_space.distributions)
        self.weights = weights
        self._distributions: Dict[str, _MixturedDistribution] = {
            name: _MixturedDistribution(
                name=name,
                distribution=distribution,
                trials=trials,
                n_distribution=n_distribution,
            )
            for name, distribution in search_space.distributions.items()
        }

    def sample_to_value(self, sample: Dict[str, Any]) -> Dict[str, ParameterValue]:
        return {
            name: self._distributions[name].sample_to_value(value)
            for name, value in sample.items()
        }

    def sample(
        self, fixed_parameters: Dict[str, ParameterValue], k: int
    ) -> Dict[str, "npt.NDArray[Any]"]:
        ret: Dict[str, "npt.NDArray[Any]"] = {}
        for name in self._search_space.distributions:
            if name in fixed_parameters:
                raise NotImplementedError()
            active = np.argmax(np.random.multinomial(1, self.weights, size=(k,)), axis=-1)
            ret[name] = self._distributions[name].sample(active_indices=active)
        return ret

    def log_pdf(self, observations: Dict[str, "npt.NDArray[Any]"]) -> "npt.NDArray[np.float64]":
        ret = np.zeros(shape=(1,))
        weights = np.log(self.weights)[None]
        for name, samples in observations.items():
            log_pdf = self._distributions[name].log_pdf(samples)
            # TODO(tsuzuku): Improve numerical stability.
            ret = ret + np.log(np.exp(log_pdf + weights).sum(axis=1))
        return ret


class _MixturedDistributionBase(abc.ABC):
    @abc.abstractclassmethod
    def sample(self, active_indices: "npt.NDArray[np.int_]") -> "npt.NDArray[Any]":
        pass

    @abc.abstractclassmethod
    def sample_to_value(self, sample: Any) -> ParameterValue:
        pass

    # (n_sample,) -> (n_sample, n_distribution)
    @abc.abstractclassmethod
    def log_pdf(self, x: "npt.NDArray[Any]") -> "npt.NDArray[np.float64]":
        pass


# TODO(tsuzuku): Implement this.
class _AitchisonAitken(_MixturedDistributionBase):
    def __init__(
        self,
        n_choice: int,
        selections: "npt.NDArray[np.int_]",
        eps: float = 1e-6,
    ) -> None:
        pass

    def sample_to_value(self, sample: Any) -> ParameterValue:
        raise NotImplementedError()

    def sample(self, active_indices: "npt.NDArray[np.int_]") -> "npt.NDArray[np.int_]":
        raise NotImplementedError()

    def log_pdf(self, x: "npt.NDArray[np.int_]") -> "npt.NDArray[np.float64]":
        raise NotImplementedError()


class _TruncatedLogisticMixturedDistribution(_MixturedDistributionBase):
    """Truncated Logistic Mixtured Distribution.

    Mimics GMM with finite support.
    We use logistic distribution instead of gaussian because logistic distribution
    is easier to implement.
    Even though scipy provides truncnorm, sometimes scipy is not easy to install and
    it's great if we can avoid introducing the dependency.

    Args:
        low:
            Lower bound of the support.
        high:
            Upper bound of the support.
        loc:
            Means of Logistic distributions.
        scale:
            Standardized variances of Logistic distributions.
        eps:
            A small constant for numerical stability.
    """

    def __init__(
        self,
        low: float,
        high: float,
        loc: "npt.NDArray[np.float64]",
        scale: "npt.NDArray[np.float64]",
        eps: float = 1e-6,
    ) -> None:
        assert loc.shape == scale.shape
        self.low = low
        self.high = high
        self.loc = loc
        self.scale = scale
        self.eps = eps
        self.normalization_constant: "npt.NDArray[np.float64]" = (  # (1, n_observation)
            self.unnormalized_cdf(np.asarray([high])) - self.unnormalized_cdf(np.asarray([low]))
        )

    def quantized_log_pdf(
        self, x: "npt.NDArray[np.float64]", q: float
    ) -> "npt.NDArray[np.float64]":
        upper_bound = np.minimum(x + q / 2.0, self.high)
        lower_bound = np.maximum(x - q / 2.0, self.low)
        probabilities: "npt.NDArray[np.float64]" = self.unnormalized_cdf(
            upper_bound
        ) - self.unnormalized_cdf(lower_bound)
        ret: "npt.NDArray[np.float64]" = np.log(probabilities + self.eps) - np.log(
            self.normalization_constant + self.eps
        )
        return ret

    def sample(self, active_indices: "npt.NDArray[np.int_]") -> "npt.NDArray[np.float64]":
        # TODO(tsuzuku): Check numerical stability.
        loc = self.loc[active_indices]
        scale = self.scale[active_indices]
        trunc_low = 1 / (1 + np.exp(-(self.low - loc) / scale))
        trunc_high = 1 / (1 + np.exp(-(self.high - loc) / scale))

        p = np.random.uniform(trunc_low, trunc_high, size=active_indices.shape)
        ret: "npt.NDArray[np.float64]" = loc - scale * np.log(1 / p - 1)
        return ret

    # Return (len(x), n_observation)
    def unnormalized_cdf(self, x: "npt.NDArray[np.float64]") -> "npt.NDArray[np.float64]":
        x = x[..., None]
        loc = self.loc[None]
        scale = self.scale[None]
        ret: "npt.NDArray[np.float64]" = 1.0 / (
            1 + np.exp(-(x - loc) / np.maximum(scale, self.eps))
        )
        return ret

    def log_pdf(self, x: "npt.NDArray[np.float64]") -> "npt.NDArray[np.float64]":
        cdf: "npt.NDArray[np.float64]" = self.unnormalized_cdf(x)
        ret: "npt.NDArray[np.float64]" = np.log(cdf * (1 - cdf)) / np.maximum(
            self.normalization_constant, self.eps
        )
        return ret

    def sample_to_value(self, sample: Any) -> ParameterValue:
        raise NotImplementedError()


class _MixturedDistribution(_MixturedDistributionBase):
    def __init__(
        self,
        name: str,
        distribution: Distribution,
        trials: Sequence[TrialProto],
        n_distribution: int,
    ) -> None:
        self._distribution = distribution
        valid_examples: "npt.NDArray[np.bool_]" = np.asarray(
            [name in trial.parameters for trial in trials]
        )
        n_observation = valid_examples.sum()
        self._kernel: _MixturedDistributionBase
        if distribution.HasField("int_distribution"):
            int_d = distribution.int_distribution
            self._kernel = self._create_numerical_distribution(
                low=float(int_d.low) - 0.5,
                high=float(int_d.high) + 0.5,
                log_scale=int_d.log_scale,
                observations=np.asarray(
                    [
                        trial.parameters[name].value.int_value if name in trial.parameters else 1.0
                        for trial in trials
                    ],
                    dtype=np.float64,
                ),
                valid=valid_examples,
                n_observation=n_observation,
                n_dimension=n_distribution,
            )
        elif distribution.HasField("float_distribution"):
            float_d = distribution.float_distribution
            self._kernel = self._create_numerical_distribution(
                low=float_d.low,
                high=float_d.high,
                log_scale=float_d.log_scale,
                observations=np.asarray(
                    [
                        trial.parameters[name].value.double_value
                        if name in trial.parameters
                        else 1.0
                        for trial in trials
                    ],
                    dtype=np.float64,
                ),
                valid=valid_examples,
                n_observation=n_observation,
                n_dimension=n_distribution,
            )
        elif distribution.HasField("categorical_distribution"):
            raise NotImplementedError(f"Unsupported distribution: {distribution}")
        elif distribution.HasField("fixed_distribution"):
            raise NotImplementedError(f"Unsupported distribution: {distribution}")
        else:
            raise NotImplementedError(f"Unsupported distribution: {distribution}")

    @staticmethod
    def _create_numerical_distribution(
        low: float,
        high: float,
        log_scale: bool,
        observations: "npt.NDArray[Any]",
        valid: "npt.NDArray[np.bool_]",
        n_observation: int,
        n_dimension: int,
    ) -> _MixturedDistributionBase:
        if log_scale:
            high = math.log(high)
            low = math.log(low)
            observations = np.log(observations) if log_scale else observations
        mus = np.where(valid, observations, np.ones_like(observations) * ((high - low) / 2))
        scales = np.where(
            valid,
            ((high - low) / 2) * (n_observation ** (-1 / (n_dimension + 4))),
            (high - low) * 100.0,  # Close to uniform distribution.
        )
        return _TruncatedLogisticMixturedDistribution(
            low=low,
            high=high,
            loc=mus,
            scale=scales,
        )

    def sample(self, active_indices: "npt.NDArray[np.int_]") -> "npt.NDArray[Any]":
        ret: "npt.NDArray[Any]"
        if self._distribution.HasField("int_distribution"):
            int_d = self._distribution.int_distribution
            samples = self._kernel.sample(active_indices=active_indices)
            if int_d.log_scale:
                samples = np.exp(samples)
            rounded_samples: "npt.NDArray[np.int_]" = np.round(  # type: ignore[no-untyped-call]
                samples
            ).astype(np.int64)
            ret = np.clip(a=rounded_samples, a_min=int_d.low, a_max=int_d.high)
            return ret
        elif self._distribution.HasField("float_distribution"):
            float_d = self._distribution.float_distribution
            samples = self._kernel.sample(active_indices=active_indices)
            if float_d.log_scale:
                samples = np.exp(samples)
            ret = np.clip(a=samples, a_min=float_d.low, a_max=float_d.high)
            return ret
        raise NotImplementedError()

    def log_pdf(self, x: "npt.NDArray[Any]") -> "npt.NDArray[np.float64]":
        if self._distribution.HasField("int_distribution"):
            int_d = self._distribution.int_distribution
            x = x.astype(np.float64)
            if int_d.log_scale:
                x = np.log(x)
            return self._kernel.log_pdf(x)
        elif self._distribution.HasField("float_distribution"):
            float_d = self._distribution.float_distribution
            if float_d.log_scale:
                x = np.log(x)
            return self._kernel.log_pdf(x)
        raise NotImplementedError()

    def sample_to_value(self, sample: Any) -> ParameterValue:
        if self._distribution.HasField("int_distribution"):
            return ParameterValue(int_value=sample)
        elif self._distribution.HasField("float_distribution"):
            return ParameterValue(double_value=sample)
        return self._kernel.sample_to_value(sample)

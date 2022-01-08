use crate::proto::optur;
use crate::samplers::random::RandomSampler;
use crate::samplers::sampler::Sampler;
use rand::distributions::WeightedIndex;
use rand::Rng;
use std::collections::HashMap;
use std::option::Option::{None, Some};
mod kernel;

pub struct TPESampler {
    fallback_sampler: RandomSampler,
}

impl TPESampler {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Sampler for TPESampler {
    fn init(&mut self, search_space: optur::SearchSpace, targets: Vec<optur::Target>) {
        self.fallback_sampler.init(search_space, targets);
    }
    fn sync(&mut self, trials: Vec<optur::Trial>) {
        self.fallback_sampler.sync(trials);
        // Trial-ID -> Idx
        // PName -> Idx -> Distribution
    }
    fn joint_sample<R: Rng + ?Sized>(
        &self,
        _fixed: &optur::Observation,
        _rng: &mut R,
    ) -> optur::Observation {
        // 1. Create weights
        // 2. Instantiate Univariate/Multivariate KDEs
        // 3. Sample from one KDE
        // 4. Compare PDF
        optur::Observation::default()
    }
    fn sample<R: Rng + ?Sized>(
        &self,
        distribution: &optur::Distribution,
        rng: &mut R,
    ) -> optur::ParameterValue {
        self.fallback_sampler.sample(&distribution, rng)
    }
}

impl Default for TPESampler {
    fn default() -> Self {
        Self {
            fallback_sampler: RandomSampler::default(),
        }
    }
}

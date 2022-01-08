use crate::proto::optur;
use crate::samplers::random::RandomSampler;
use crate::samplers::sampler::Sampler;
use crate::search_space::tracker::SearchSpaceTracker;
use rand::distributions::WeightedIndex;
use rand::Rng;
use std::collections::HashMap;
use std::option::Option::{None, Some};
mod kernel;
use kernel::logit::LogitKernels;
use kernel::UnivariateKernel;

pub struct TPESampler {
    fallback_sampler: RandomSampler,
    search_space_tracker: SearchSpaceTracker,
    trial_id_to_idx: HashMap<String, usize>,
    int_kernels: HashMap<String, LogitKernels>,
    double_kernels: HashMap<String, LogitKernels>,
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
    fn sync(&mut self, trials: &Vec<optur::Trial>) {
        self.fallback_sampler.sync(trials);
        for trial in trials {
            if self.trial_id_to_idx.contains_key(&trial.trial_id) {
                // Update.
            } else {
                // Insert.
                let idx = self.trial_id_to_idx.len()
                self.trial_id_to_idx.insert(trial.trial_id.clone(), self.trial_id_to_idx.len());
            }
        }
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
            search_space_tracker: SearchSpaceTracker::default(),
            trial_id_to_idx: HashMap::new(),
            int_kernels: HashMap::new(),
            double_kernels: HashMap::new(),
        }
    }
}

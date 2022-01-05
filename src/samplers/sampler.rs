use crate::proto::optur;
use rand::Rng;

pub trait Sampler {
    /// Initialize or reset the internal states.
    fn init(&mut self, search_space: optur::SearchSpace, targets: Vec<optur::Target>);

    /// Read trials and update the internal caches.
    fn sync(&mut self, trials: Vec<optur::Trial>);

    /// Suggest parameters from an estimated search space.
    fn joint_sample<R: Rng + ?Sized>(
        &self,
        fixed: &optur::Observation,
        rng: &mut R,
    ) -> optur::Observation;

    /// Suggest a parameter from the given distribution.
    fn sample<R: Rng + ?Sized>(
        &self,
        distribution: &optur::Distribution,
        rng: &mut R,
    ) -> optur::ParameterValue;
}

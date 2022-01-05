use crate::proto::optur;
use rand::Rng;

pub trait Sampler {
    fn init(&self, search_space: optur::SearchSpace, targets: Vec<optur::Target>);
    fn sync(&self, trials: Vec<optur::Trial>);
    fn joint_sample<R: Rng + ?Sized>(
        &self,
        fixed: optur::Observation,
        rng: &mut R,
    ) -> optur::Observation;
    fn sample<R: Rng + ?Sized>(
        &self,
        distribution: optur::Distribution,
        rng: &mut R,
    ) -> optur::ParameterValue;
}

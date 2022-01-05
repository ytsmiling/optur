use crate::proto::optur;
use crate::samplers::sampler::Sampler;
use rand::Rng;
use std::option::Option::{None, Some};

pub struct RandomSampler {}

impl RandomSampler {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Sampler for RandomSampler {
    fn init(&mut self, _search_space: optur::SearchSpace, _targets: Vec<optur::Target>) {}
    fn sync(&mut self, _trials: Vec<optur::Trial>) {}
    fn joint_sample<R: Rng + ?Sized>(
        &self,
        _fixed: optur::Observation,
        _rng: &mut R,
    ) -> optur::Observation {
        optur::Observation::default()
    }
    fn sample<R: Rng + ?Sized>(
        &self,
        distribution: optur::Distribution,
        rng: &mut R,
    ) -> optur::ParameterValue {
        match distribution.distribution {
            Some(optur::distribution::Distribution::UnknownDistribution(_)) => {
                return optur::ParameterValue { value: None }
            }
            Some(optur::distribution::Distribution::IntDistribution(int_d)) => {
                return optur::ParameterValue {
                    value: Some(optur::parameter_value::Value::IntValue(
                        rng.gen_range(int_d.low..int_d.high + 1),
                    )),
                }
            }
            Some(optur::distribution::Distribution::FloatDistribution(float_d)) => {
                return optur::ParameterValue {
                    value: Some(optur::parameter_value::Value::DoubleValue(
                        rng.gen::<f64>() * (float_d.high - float_d.low) + float_d.low,
                    )),
                }
            }
            Some(optur::distribution::Distribution::CategoricalDistribution(_cat_d)) => {
                panic!()
            }
            Some(optur::distribution::Distribution::FixedDistribution(fix_d)) => {
                return match fix_d.value {
                    Some(v) => v,
                    None => {
                        panic!()
                    }
                }
            }
            None => {
                panic!()
            }
        }
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn init_works() {
        let sampler = RandomSampler::new();
        let search_space = optur::SearchSpace::default();
        let targets = Vec::<optur::Target>::new();
        sampler.init(search_space, targets);
    }

    #[test]
    fn sync_works() {
        let sampler = RandomSampler::new();
        sampler.sync(Vec::<optur::Trial>::new());
    }
}

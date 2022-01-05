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
        _fixed: &optur::Observation,
        _rng: &mut R,
    ) -> optur::Observation {
        optur::Observation::default()
    }
    fn sample<R: Rng + ?Sized>(
        &self,
        distribution: &optur::Distribution,
        rng: &mut R,
    ) -> optur::ParameterValue {
        match &distribution.distribution {
            Some(optur::distribution::Distribution::UnknownDistribution(_)) => {
                optur::ParameterValue { value: None }
            }
            Some(optur::distribution::Distribution::IntDistribution(int_d)) => {
                optur::ParameterValue {
                    value: Some(optur::parameter_value::Value::IntValue(
                        rng.gen_range(int_d.low..int_d.high + 1),
                    )),
                }
            }
            Some(optur::distribution::Distribution::FloatDistribution(float_d)) => {
                optur::ParameterValue {
                    value: Some(optur::parameter_value::Value::DoubleValue(
                        rng.gen::<f64>() * (float_d.high - float_d.low) + float_d.low,
                    )),
                }
            }
            Some(optur::distribution::Distribution::CategoricalDistribution(cat_d)) => {
                let idx = rng.gen_range(0..cat_d.choices.len());
                cat_d.choices[idx].clone()
            }
            Some(optur::distribution::Distribution::FixedDistribution(fix_d)) => {
                match &fix_d.value {
                    Some(v) => v.clone(),
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

    #[test]
    fn init_works() {
        let mut sampler = RandomSampler::new();
        let search_space = optur::SearchSpace::default();
        let targets = Vec::<optur::Target>::new();
        sampler.init(search_space, targets);
    }

    #[test]
    fn sync_works() {
        let mut sampler = RandomSampler::new();
        sampler.sync(Vec::<optur::Trial>::new());
    }

    #[test]
    fn joint_sample_works() {
        let sampler = RandomSampler::new();
        let mut rng = rand::thread_rng();
        let fixed = optur::Observation::default();
        sampler.joint_sample(&fixed, &mut rng);
    }

    #[test]
    fn sample_int_works() {
        let sampler = RandomSampler::new();
        let mut rng = rand::thread_rng();
        let dist = optur::Distribution {
            distribution: Some(optur::distribution::Distribution::IntDistribution(
                optur::distribution::IntDistribution {
                    low: 1,
                    high: 5,
                    log_scale: false,
                },
            )),
        };
        for _i in 0..100 {
            let v = sampler.sample(&dist, &mut rng);
            match v.value {
                Some(optur::parameter_value::Value::IntValue(i)) => {
                    assert!(1 <= i && i <= 5)
                }
                _ => {
                    assert!(false)
                }
            }
        }
    }
}

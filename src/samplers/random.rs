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

    #[test]
    fn sample_float_works() {
        let sampler = RandomSampler::new();
        let mut rng = rand::thread_rng();
        let dist = optur::Distribution {
            distribution: Some(optur::distribution::Distribution::FloatDistribution(
                optur::distribution::FloatDistribution {
                    low: 1.0,
                    high: 5.0,
                    log_scale: false,
                },
            )),
        };
        for _i in 0..100 {
            let v = sampler.sample(&dist, &mut rng);
            match v.value {
                Some(optur::parameter_value::Value::DoubleValue(i)) => {
                    assert!(1.0 <= i && i <= 5.0)
                }
                _ => {
                    assert!(false)
                }
            }
        }
    }

    #[test]
    fn sample_categorical_works() {
        let sampler = RandomSampler::new();
        let mut rng = rand::thread_rng();
        let dist = optur::Distribution {
            distribution: Some(optur::distribution::Distribution::CategoricalDistribution(
                optur::distribution::CategoricalDistribution {
                    choices: vec![
                        optur::ParameterValue {
                            value: Some(optur::parameter_value::Value::IntValue(2)),
                        },
                        optur::ParameterValue {
                            value: Some(optur::parameter_value::Value::DoubleValue(0.2)),
                        },
                    ],
                },
            )),
        };
        for _i in 0..100 {
            let v = sampler.sample(&dist, &mut rng);
            match v.value {
                Some(optur::parameter_value::Value::IntValue(i)) => {
                    assert!(i == 2)
                }
                Some(optur::parameter_value::Value::DoubleValue(i)) => {
                    assert!(i == 0.2)
                }
                _ => {
                    assert!(false)
                }
            }
        }
    }
}

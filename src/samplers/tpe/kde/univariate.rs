use crate::proto::optur;
use crate::samplers::tpe::kernel::logit;
use crate::samplers::tpe::kernel::logit::LogitKernels;
use crate::search_space::distribution::{double_value, int_value};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;
use std::collections::HashMap;

pub struct UnivariateKDE {
    trial_id_to_idx: HashMap<String, usize>,
    int_kernels: HashMap<String, LogitKernels>,
    double_kernels: HashMap<String, LogitKernels>,
}

impl UnivariateKDE {
    pub fn default() -> UnivariateKDE {
        UnivariateKDE {
            trial_id_to_idx: HashMap::new(),
            int_kernels: HashMap::new(),
            double_kernels: HashMap::new(),
        }
    }
    pub fn sync(&mut self, trials: &Vec<optur::Trial>) {}
    pub fn init(&self, weights: &HashMap<String, f64>) -> UnivariateKDESampler {
        let mut w = vec![0.0; self.trial_id_to_idx.len()];
        for (key, value) in weights {
            w[self.trial_id_to_idx[key]] = *value;
        }
        UnivariateKDESampler {
            int_kernels: &self.int_kernels,
            double_kernels: &self.double_kernels,
            weights: w,
        }
    }
}

pub struct UnivariateKDESampler<'a> {
    int_kernels: &'a HashMap<String, LogitKernels>,
    double_kernels: &'a HashMap<String, LogitKernels>,
    weights: Vec<f64>,
}

impl UnivariateKDESampler<'_> {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> optur::Observation {
        let mut observation = optur::Observation::default();
        for (key, kernel) in self.int_kernels {
            let active_idx = WeightedIndex::new(&self.weights).unwrap().sample(rng);
            observation.parameters.insert(
                key.clone(),
                int_value(kernel.sample(rng, active_idx) as i64),
            );
        }
        for (key, kernel) in self.double_kernels {
            let active_idx = WeightedIndex::new(&self.weights).unwrap().sample(rng);
            observation
                .parameters
                .insert(key.clone(), double_value(kernel.sample(rng, active_idx)));
        }
        observation
    }

    pub fn log_pdf(&self, observation: &optur::Observation) -> f64 {
        let mut ret = 0.0;
        for (key, kernel) in self.int_kernels {
            match observation.parameters[key].value {
                Some(optur::parameter_value::Value::IntValue(v)) => {
                    ret += kernel.log_pdf(v as f64).iter().sum::<f64>();
                }
                _ => {
                    panic!();
                }
            }
        }
        for (key, kernel) in self.double_kernels {
            match observation.parameters[key].value {
                Some(optur::parameter_value::Value::DoubleValue(v)) => {
                    ret += kernel.log_pdf(v).iter().sum::<f64>();
                }
                _ => {
                    panic!();
                }
            }
        }
        ret
    }
}

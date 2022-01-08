mod logit;

use crate::proto::optur;
use crate::samplers::tpe::kernel::logit::LogitKernels;
use crate::search_space::distribution::{double_value, int_value};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;
use std::collections::HashMap;

pub struct UnivariateKernel<'a> {
    int_kernels: &'a HashMap<String, LogitKernels>,
    double_kernels: &'a HashMap<String, LogitKernels>,
}

impl UnivariateKernel<'_> {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R, weights: &Vec<f64>) -> optur::Observation {
        let mut observation = optur::Observation::default();
        for (key, kernel) in self.int_kernels {
            let active_idx = WeightedIndex::new(weights).unwrap().sample(rng);
            observation.parameters.insert(
                key.clone(),
                int_value(kernel.sample(rng, active_idx) as i64),
            );
        }
        for (key, kernel) in self.double_kernels {
            let active_idx = WeightedIndex::new(weights).unwrap().sample(rng);
            observation
                .parameters
                .insert(key.clone(), double_value(kernel.sample(rng, active_idx)));
        }
        observation
    }

    pub fn log_pdf(&self, observation: &optur::Observation, weights: &Vec<f64>) -> f64 {
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

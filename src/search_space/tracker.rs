use crate::proto::optur;
use crate::proto::optur::distribution::Distribution::{
    CategoricalDistribution, UnknownDistribution,
};
use crate::search_space::distribution::unknown_distribution;
use std::collections::HashSet;
use std::option::Option::{None, Some};

struct SearchSpaceTracker {
    search_space: optur::SearchSpace,
}

fn convert_to_hash_set(
    a: &Vec<optur::ParameterValue>,
) -> (HashSet<&i64>, HashSet<[u8; 8]>, HashSet<&String>) {
    let mut int_set = HashSet::new();
    let mut double_set = HashSet::new();
    let mut string_set = HashSet::new();
    for p in a.iter() {
        match &p.value {
            Some(optur::parameter_value::Value::IntValue(v)) => {
                int_set.insert(v);
            }
            Some(optur::parameter_value::Value::DoubleValue(v)) => {
                // TODO(tsuzuku): Check that v is not `nan` or `inf`.
                double_set.insert(v.to_ne_bytes());
            }
            Some(optur::parameter_value::Value::StringValue(v)) => {
                string_set.insert(v);
            }
            None => {
                panic!();
            }
        }
    }
    (int_set, double_set, string_set)
}

fn is_equal_parameter_set(a: &Vec<optur::ParameterValue>, b: &Vec<optur::ParameterValue>) -> bool {
    convert_to_hash_set(a) == convert_to_hash_set(b)
}

impl SearchSpaceTracker {
    fn sync(&mut self, trials: &Vec<optur::Trial>) {
        for trial in trials.iter() {
            for (name, parameter) in &trial.parameters {
                if self.search_space.distributions.contains_key(name) {
                    let old_dist = &self.search_space.distributions[name];
                    match &parameter.distribution {
                        Some(dist) => {
                            match (dist.distribution.as_ref(), old_dist.distribution.as_ref()) {
                                (Some(UnknownDistribution(_)), Some(UnknownDistribution(_))) => {
                                    self.search_space
                                        .distributions
                                        .insert(name.to_string(), dist.clone());
                                }
                                (Some(UnknownDistribution(_)), b) => {}
                                (a, Some(UnknownDistribution(_))) => {}
                                (
                                    Some(CategoricalDistribution(a)),
                                    Some(CategoricalDistribution(b)),
                                ) => {
                                    assert!(is_equal_parameter_set(&a.choices, &b.choices));
                                }
                                (a, b) => {
                                    assert!(a == b);
                                }
                            }
                        }
                        None => {
                            match self.search_space.distributions[name].distribution.as_ref() {
                                Some(UnknownDistribution(_)) => {
                                    panic!()
                                } // TODO(tsuzuku): Update.
                                Some(_) => {
                                    panic!()
                                } // TODO(tsuzuku): Check the value in the dist.
                                None => {
                                    panic!()
                                }
                            }
                        }
                    };
                } else {
                    self.search_space.distributions.insert(
                        name.to_string(),
                        unknown_distribution(parameter.value.as_ref().unwrap().clone()),
                    );
                }
            }
        }
    }
}

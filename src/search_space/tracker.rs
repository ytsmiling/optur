use crate::proto::optur;
use crate::proto::optur::distribution::Distribution::{
    CategoricalDistribution, UnknownDistribution,
};
use crate::search_space::distribution::{contains, unknown_distribution};
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

fn extend_parameter_set(a: &mut Vec<optur::ParameterValue>, b: &Vec<optur::ParameterValue>) {
    let mut int_set = HashSet::<&i64>::new();
    let mut double_set = HashSet::<[u8; 8]>::new();
    let mut string_set = HashSet::<&String>::new();
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
    let mut value_to_extend = Vec::<&optur::ParameterValue>::new();
    for p in b.iter() {
        match &p.value {
            Some(optur::parameter_value::Value::IntValue(v)) => {
                if !int_set.contains(v) {
                    value_to_extend.push(p)
                }
            }
            Some(optur::parameter_value::Value::DoubleValue(v)) => {
                // TODO(tsuzuku): Check that v is not `nan` or `inf`.
                if !double_set.contains(&v.to_ne_bytes()) {
                    value_to_extend.push(p)
                }
            }
            Some(optur::parameter_value::Value::StringValue(v)) => {
                if !string_set.contains(v) {
                    value_to_extend.push(p)
                }
            }
            None => {
                panic!();
            }
        }
    }
    for v in value_to_extend {
        a.push(v.clone());
    }
}

impl SearchSpaceTracker {
    fn sync(&mut self, trials: &Vec<optur::Trial>) {
        for trial in trials.iter() {
            for (name, parameter) in &trial.parameters {
                if self.search_space.distributions.contains_key(name) {
                    let mut old_dist = self.search_space.distributions.get_mut(name).unwrap();
                    match &parameter.distribution {
                        Some(dist) => {
                            match (dist.distribution.as_ref(), old_dist.distribution.as_mut()) {
                                (Some(UnknownDistribution(nd)), Some(UnknownDistribution(od))) => {
                                    extend_parameter_set(&mut od.values, &nd.values);
                                }
                                (Some(UnknownDistribution(nd)), _) => {
                                    assert!(nd.values.iter().any(|c| contains(&old_dist, &c)));
                                }
                                (_, Some(UnknownDistribution(od))) => {
                                    assert!(od.values.iter().any(|c| contains(&dist, &c)));
                                }
                                (
                                    Some(CategoricalDistribution(a)),
                                    Some(CategoricalDistribution(b)),
                                ) => {
                                    assert!(is_equal_parameter_set(&a.choices, &b.choices));
                                }
                                (a, b) => {
                                    assert!(a.unwrap() == b.unwrap());
                                }
                            }
                        }
                        None => match old_dist.distribution.as_mut() {
                            Some(UnknownDistribution(od)) => {
                                let v = vec![parameter.value.as_ref().unwrap().clone()];
                                extend_parameter_set(&mut od.values, &v);
                            }
                            Some(dist) => {
                                assert!(contains(&old_dist, &parameter.value.as_ref().unwrap()))
                            }
                            None => {
                                panic!()
                            }
                        },
                    };
                } else {
                    self.search_space.distributions.insert(
                        name.to_string(),
                        unknown_distribution(vec![parameter.value.as_ref().unwrap().clone()]),
                    );
                }
            }
        }
    }
}

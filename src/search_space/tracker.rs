use crate::proto::optur;
use crate::proto::optur::distribution::Distribution::{
    CategoricalDistribution, UnknownDistribution,
};
use crate::search_space::distribution::unknown_distribution;
use std::option::Option::{None, Some};

struct SearchSpaceTracker {
    search_space: optur::SearchSpace,
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
                                    assert!(a == b); // TODO(tsuzuku): Compare the set of choices.
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

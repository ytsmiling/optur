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
                    let new_dist = match &parameter.distribution {
                        Some(dist) => {
                            match (
                                dist.distribution.as_ref(),
                                self.search_space.distributions[name].distribution.as_ref(),
                            ) {
                                (Some(UnknownDistribution(_)), Some(UnknownDistribution(_))) => {
                                    dist.distribution.as_ref() // TODO(tsuzuku): Merge the two unknowns.
                                }
                                (Some(UnknownDistribution(_)), b) => b,
                                (a, Some(UnknownDistribution(_))) => a,
                                (
                                    Some(CategoricalDistribution(a)),
                                    Some(CategoricalDistribution(b)),
                                ) => {
                                    assert!(a == b); // TODO(tsuzuku): Compare the set of choices.
                                    dist.distribution.as_ref()
                                }
                                (a, b) => {
                                    assert!(a == b);
                                    a
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
                    let new_dist = new_dist.unwrap().clone();
                    self.search_space.distributions.insert(
                        name.to_string(),
                        optur::Distribution {
                            distribution: Some(new_dist),
                        },
                    );
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

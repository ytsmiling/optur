use crate::proto::optur;
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
                    // TODO(tsuzuku): update distribution.
                    match &parameter.distribution {
                        Some(dist) => {} // Update distribution.
                        None => {}       // Treat like unknown distribution.
                    }
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

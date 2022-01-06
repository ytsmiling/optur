use crate::proto::optur;

pub fn int_distribution(low: i64, high: i64, log_scale: bool) -> optur::Distribution {
    optur::Distribution {
        distribution: Some(optur::distribution::Distribution::IntDistribution(
            optur::distribution::IntDistribution {
                low,
                high,
                log_scale,
            },
        )),
    }
}

pub fn float_distribution(low: f64, high: f64, log_scale: bool) -> optur::Distribution {
    optur::Distribution {
        distribution: Some(optur::distribution::Distribution::FloatDistribution(
            optur::distribution::FloatDistribution {
                low,
                high,
                log_scale,
            },
        )),
    }
}

pub fn categorical_distribution(choices: Vec<optur::ParameterValue>) -> optur::Distribution {
    optur::Distribution {
        distribution: Some(optur::distribution::Distribution::CategoricalDistribution(
            optur::distribution::CategoricalDistribution { choices },
        )),
    }
}

pub fn fixed_distribution(value: optur::ParameterValue) -> optur::Distribution {
    optur::Distribution {
        distribution: Some(optur::distribution::Distribution::FixedDistribution(
            optur::distribution::FixedDistribution { value: Some(value) },
        )),
    }
}

pub fn unknown_distribution(value: optur::ParameterValue) -> optur::Distribution {
    optur::Distribution {
        distribution: Some(optur::distribution::Distribution::UnknownDistribution(
            optur::distribution::UnknownDistribution {
                values: vec![value],
            },
        )),
    }
}

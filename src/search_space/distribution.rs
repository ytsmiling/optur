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

pub fn unknown_distribution(values: Vec<optur::ParameterValue>) -> optur::Distribution {
    optur::Distribution {
        distribution: Some(optur::distribution::Distribution::UnknownDistribution(
            optur::distribution::UnknownDistribution { values },
        )),
    }
}

pub fn contains(distribution: &optur::Distribution, parameter: &optur::ParameterValue) -> bool {
    match (
        &distribution.distribution.as_ref(),
        &parameter.value.as_ref(),
    ) {
        (Some(optur::distribution::Distribution::UnknownDistribution(dist)), _) => {
            panic!(); // Undefined.
        }
        (
            Some(optur::distribution::Distribution::IntDistribution(dist)),
            Some(optur::parameter_value::Value::IntValue(i)),
        ) => dist.low <= *i && *i <= dist.high,
        (
            Some(optur::distribution::Distribution::FloatDistribution(dist)),
            Some(optur::parameter_value::Value::DoubleValue(i)),
        ) => dist.low <= *i && *i <= dist.high,
        (Some(optur::distribution::Distribution::CategoricalDistribution(dist)), Some(p)) => {
            dist.choices.iter().any(|c| &c.value.as_ref().unwrap() == p)
        }
        (Some(optur::distribution::Distribution::FixedDistribution(dist)), Some(p)) => {
            &dist.value.as_ref().unwrap().value.as_ref().unwrap() == p
        }
        _ => false,
    }
}

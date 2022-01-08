use crate::samplers::random::RandomSampler;
use crate::samplers::sampler::Sampler;
use rand::Rng;
use std::collections::HashMap;
use std::option::Option::{None, Some};

const EPS: f64 = 1e-6;

pub struct LogitKernels {
    low: f64,
    high: f64,
    kernels: Vec<LogitKernel>,
}

impl LogitKernels {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R, index: usize) -> f64 {
        self.kernels[index].sample(rng, self.low, self.high)
    }

    pub fn log_pdf(&self, value: f64) -> Vec<f64> {
        let mut out = Vec::new();
        out.reserve(self.kernels.len());
        out.extend(
            self.kernels
                .iter()
                .map(|k| k.log_pdf(value, self.low, self.high)),
        );
        out
    }
}

pub struct LogitKernel {
    mean: f64,
    scale: f64,
}

impl LogitKernel {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, low: f64, high: f64) -> f64 {
        sample(rng, low, high, self.mean, self.scale)
    }

    fn log_pdf(&self, value: f64, low: f64, high: f64) -> f64 {
        log_pdf(value, self.mean, self.scale, low, high)
    }
}

fn sample<R: Rng + ?Sized>(rng: &mut R, low: f64, high: f64, mean: f64, scale: f64) -> f64 {
    let trunc_low = 1.0 / (1.0 + (-(low - mean) / scale).exp());
    let trunc_high = 1.0 / (1.0 + (-(high - mean) / scale).exp());
    let p = rng.gen_range(trunc_low..trunc_high);
    mean - scale * (1.0 / p - 1.0).ln()
}

fn unnormalized_cdf(x: f64, mean: f64, scale: f64) -> f64 {
    1.0 / (1.0 + (-(x - mean) / (scale + EPS)).exp())
}

fn log_pdf(x: f64, mean: f64, scale: f64, low: f64, high: f64) -> f64 {
    let cdf = unnormalized_cdf(x, mean, scale);
    let normalization_constant =
        unnormalized_cdf(high, mean, scale) - unnormalized_cdf(low, mean, scale);
    (cdf * (1.0 - cdf)).ln() / (normalization_constant + EPS)
}

use std::time::Instant;

use candelora_metal_kernels::activation_quant::{ActivationQuantConfig, ActivationQuantMode};
use candelora_metal_kernels::iq2_matmul::{iq2_matmul_with_activation_quant, Iq2MatmulVariant};
use candle_core::{DType, Device, Result, Tensor};

const IQ_QK: usize = 256;

#[derive(Debug, Clone)]
struct Config {
    variants: Vec<Iq2MatmulVariant>,
    activation_quant_mode: ActivationQuantMode,
    m: usize,
    out_dim: usize,
    in_dim: usize,
    warmup_runs: usize,
    runs: usize,
    dtype: DType,
}

#[derive(Debug, Clone)]
struct VariantSummary {
    label: &'static str,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    matmuls_per_s: f64,
}

fn print_usage() {
    println!(
        "Usage: cargo run --release --features metal --example iq2_matmul_bench -- [options]

Options:
  --variants <csv>       Comma list: iq2-xxs,iq2-xs,iq2-s,iq3-s
                         default: iq2-xxs,iq2-xs,iq2-s,iq3-s
  --activation-quant-mode <off|w8a8>
                         Activation quant mode (default: off)
  --m <n>                Batch rows for x (default: 1)
  --out-dim <n>          Output dimension (default: 4096)
  --in-dim <n>           Input dimension (default: 4096)
  --warmup-runs <n>      Warmup runs per variant (default: 5)
  --runs <n>             Measured runs per variant (default: 20)
  --dtype <f16|bf16|f32> Input dtype for x (default: f16)
  --help                 Show this help"
    );
}

fn parse_usize(flag: &str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|e| candle_core::Error::msg(format!("invalid {} `{}`: {}", flag, value, e)))
}

fn parse_dtype(value: &str) -> Result<DType> {
    match value {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        other => candle_core::bail!("invalid --dtype `{}` (expected f16|bf16|f32)", other),
    }
}

fn parse_variant(value: &str) -> Result<Iq2MatmulVariant> {
    match value {
        "iq2-xxs" => Ok(Iq2MatmulVariant::Iq2Xxs),
        "iq2-xs" => Ok(Iq2MatmulVariant::Iq2Xs),
        "iq2-s" => Ok(Iq2MatmulVariant::Iq2S),
        "iq3-s" => Ok(Iq2MatmulVariant::Iq3S),
        other => candle_core::bail!(
            "invalid variant `{}` (expected iq2-xxs|iq2-xs|iq2-s|iq3-s)",
            other
        ),
    }
}

fn parse_activation_quant_mode(value: &str) -> Result<ActivationQuantMode> {
    match value {
        "off" => Ok(ActivationQuantMode::Off),
        "w8a8" => Ok(ActivationQuantMode::W8A8),
        other => candle_core::bail!(
            "invalid --activation-quant-mode `{}` (expected off|w8a8)",
            other
        ),
    }
}

fn parse_variants(csv: &str) -> Result<Vec<Iq2MatmulVariant>> {
    let mut out = Vec::new();
    for raw in csv.split(',') {
        let v = raw.trim();
        if v.is_empty() {
            continue;
        }
        out.push(parse_variant(v)?);
    }
    if out.is_empty() {
        candle_core::bail!("--variants resolved to an empty list");
    }
    Ok(out)
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config {
        variants: vec![
            Iq2MatmulVariant::Iq2Xxs,
            Iq2MatmulVariant::Iq2Xs,
            Iq2MatmulVariant::Iq2S,
            Iq2MatmulVariant::Iq3S,
        ],
        activation_quant_mode: ActivationQuantMode::Off,
        m: 1,
        out_dim: 4096,
        in_dim: 4096,
        warmup_runs: 5,
        runs: 20,
        dtype: DType::F16,
    };

    let mut args = std::env::args().skip(1).peekable();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--variants" => {
                let value = args
                    .next()
                    .ok_or_else(|| candle_core::Error::msg("missing value for --variants"))?;
                cfg.variants = parse_variants(&value)?;
            }
            "--activation-quant-mode" => {
                let value = args.next().ok_or_else(|| {
                    candle_core::Error::msg("missing value for --activation-quant-mode")
                })?;
                cfg.activation_quant_mode = parse_activation_quant_mode(&value)?;
            }
            "--m" => {
                let value = args
                    .next()
                    .ok_or_else(|| candle_core::Error::msg("missing value for --m"))?;
                cfg.m = parse_usize("--m", &value)?;
            }
            "--out-dim" => {
                let value = args
                    .next()
                    .ok_or_else(|| candle_core::Error::msg("missing value for --out-dim"))?;
                cfg.out_dim = parse_usize("--out-dim", &value)?;
            }
            "--in-dim" => {
                let value = args
                    .next()
                    .ok_or_else(|| candle_core::Error::msg("missing value for --in-dim"))?;
                cfg.in_dim = parse_usize("--in-dim", &value)?;
            }
            "--warmup-runs" => {
                let value = args
                    .next()
                    .ok_or_else(|| candle_core::Error::msg("missing value for --warmup-runs"))?;
                cfg.warmup_runs = parse_usize("--warmup-runs", &value)?;
            }
            "--runs" => {
                let value = args
                    .next()
                    .ok_or_else(|| candle_core::Error::msg("missing value for --runs"))?;
                cfg.runs = parse_usize("--runs", &value)?;
            }
            "--dtype" => {
                let value = args
                    .next()
                    .ok_or_else(|| candle_core::Error::msg("missing value for --dtype"))?;
                cfg.dtype = parse_dtype(&value)?;
            }
            other => candle_core::bail!("unknown option `{}`", other),
        }
    }

    if cfg.m == 0 || cfg.out_dim == 0 || cfg.in_dim == 0 {
        candle_core::bail!("--m, --out-dim, and --in-dim must all be > 0");
    }
    if cfg.runs == 0 {
        candle_core::bail!("--runs must be > 0");
    }
    Ok(cfg)
}

fn variant_label(variant: Iq2MatmulVariant) -> &'static str {
    match variant {
        Iq2MatmulVariant::Iq2Xxs => "iq2-xxs",
        Iq2MatmulVariant::Iq2Xs => "iq2-xs",
        Iq2MatmulVariant::Iq2S => "iq2-s",
        Iq2MatmulVariant::Iq3S => "iq3-s",
    }
}

fn bytes_per_block(variant: Iq2MatmulVariant) -> usize {
    match variant {
        Iq2MatmulVariant::Iq2Xxs => 64,
        Iq2MatmulVariant::Iq2Xs => 64,
        Iq2MatmulVariant::Iq2S => 72,
        Iq2MatmulVariant::Iq3S => 104,
    }
}

fn scales_per_block(variant: Iq2MatmulVariant) -> usize {
    match variant {
        Iq2MatmulVariant::Iq2Xxs => 2,
        Iq2MatmulVariant::Iq2Xs => 10,
        Iq2MatmulVariant::Iq2S => 10,
        Iq2MatmulVariant::Iq3S => 6,
    }
}

fn lcg_fill(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut out = vec![0u8; len];
    for byte in &mut out {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    out
}

fn force_f16_one_per_block(scales: &mut [u8], scales_per_block: usize) {
    for block in scales.chunks_exact_mut(scales_per_block) {
        block[0] = 0x00;
        block[1] = 0x3c;
    }
}

fn deterministic_x(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17) % 41) as f32 - 20.0) / 13.0)
        .collect()
}

fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    let rank = ((p * n as f64).ceil() as usize).saturating_sub(1);
    values[rank.min(n - 1)]
}

fn summarize(values: &[f64]) -> (f64, f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let p50 = percentile(values.to_vec(), 0.50);
    let p95 = percentile(values.to_vec(), 0.95);
    (mean, p50, p95)
}

fn bench_variant(
    device: &Device,
    cfg: &Config,
    variant: Iq2MatmulVariant,
) -> Result<VariantSummary> {
    let blocks_per_row = cfg.in_dim.div_ceil(IQ_QK);
    let num_blocks = cfg.out_dim * blocks_per_row;
    let mut weight_bytes = lcg_fill(
        num_blocks * bytes_per_block(variant),
        0xA11CE5ED ^ (cfg.out_dim as u32),
    );
    let mut weight_scales = lcg_fill(
        num_blocks * scales_per_block(variant),
        0xC0FFEE12 ^ (cfg.in_dim as u32),
    );
    force_f16_one_per_block(&mut weight_scales, scales_per_block(variant));
    if let Some(last) = weight_bytes.last_mut() {
        *last = 0x5A;
    }

    let x_host = deterministic_x(cfg.m * cfg.in_dim);
    let x = Tensor::from_slice(&x_host, (cfg.m, cfg.in_dim), device)?.to_dtype(cfg.dtype)?;
    let weight_bytes = Tensor::from_slice(&weight_bytes, weight_bytes.len(), device)?;
    let weight_scales = Tensor::from_slice(&weight_scales, weight_scales.len(), device)?;

    for _ in 0..cfg.warmup_runs {
        let _ = iq2_matmul_with_activation_quant(
            &x,
            &weight_bytes,
            &weight_scales,
            cfg.out_dim,
            cfg.in_dim,
            variant,
            &ActivationQuantConfig {
                mode: cfg.activation_quant_mode,
                strict: false,
            },
        )?;
        device.synchronize()?;
    }

    let mut timings_ms = Vec::with_capacity(cfg.runs);
    for _ in 0..cfg.runs {
        let t0 = Instant::now();
        let y = iq2_matmul_with_activation_quant(
            &x,
            &weight_bytes,
            &weight_scales,
            cfg.out_dim,
            cfg.in_dim,
            variant,
            &ActivationQuantConfig {
                mode: cfg.activation_quant_mode,
                strict: false,
            },
        )?;
        device.synchronize()?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
        timings_ms.push(elapsed_ms);
        let _ = y;
    }

    let (mean_ms, p50_ms, p95_ms) = summarize(&timings_ms);
    let matmuls_per_s = if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 };

    Ok(VariantSummary {
        label: variant_label(variant),
        mean_ms,
        p50_ms,
        p95_ms,
        matmuls_per_s,
    })
}

#[cfg(feature = "metal")]
fn run() -> Result<()> {
    let cfg = parse_args()?;
    let device = Device::metal_if_available(0)?;
    if !device.is_metal() {
        candle_core::bail!("Metal device not available");
    }

    println!(
        "[iq2-bench] device={:?} dtype={:?} act_quant={:?} shape=({}, {}, {}) warmup_runs={} runs={}",
        device,
        cfg.dtype,
        cfg.activation_quant_mode,
        cfg.m,
        cfg.out_dim,
        cfg.in_dim,
        cfg.warmup_runs,
        cfg.runs
    );
    println!("[iq2-bench] variants={:?}", cfg.variants);
    println!("variant\tmean_ms\tp50_ms\tp95_ms\tmatmuls_per_s");
    for variant in &cfg.variants {
        let stats = bench_variant(&device, &cfg, *variant)?;
        println!(
            "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            stats.label, stats.mean_ms, stats.p50_ms, stats.p95_ms, stats.matmuls_per_s
        );
    }
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn run() -> Result<()> {
    candle_core::bail!("iq2_matmul_bench requires --features metal")
}

fn main() -> Result<()> {
    run()
}

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

VARIANTS="iq2-xxs,iq2-xs,iq2-s,iq3-s"
M=1
OUT_DIM=4096
IN_DIM=4096
WARMUP_RUNS=5
RUNS=20
DTYPE="f16"
SKIP_PARITY=0
SKIP_BENCH=0

usage() {
  cat <<'USAGE'
Usage: bash run_iq_parity_bench.sh [options]

Runs:
  1) CPU-vs-Metal parity tests per IQ variant (from src/iq2_matmul.rs tests)
  2) Kernel microbench with mean/p50/p95 latency per variant

Options:
  --variants <csv>          Comma list: iq2-xxs,iq2-xs,iq2-s,iq3-s
                            default: iq2-xxs,iq2-xs,iq2-s,iq3-s
  --m <n>                   Batch rows for x in benchmark (default: 1)
  --out-dim <n>             Output dimension (default: 4096)
  --in-dim <n>              Input dimension (default: 4096)
  --warmup-runs <n>         Warmup runs for benchmark (default: 5)
  --runs <n>                Measured runs for benchmark (default: 20)
  --dtype <f16|bf16|f32>    Input dtype for benchmark x (default: f16)
  --skip-parity             Skip parity tests
  --skip-bench              Skip benchmark example
  --help                    Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variants) VARIANTS="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --out-dim) OUT_DIM="$2"; shift 2 ;;
    --in-dim) IN_DIM="$2"; shift 2 ;;
    --warmup-runs) WARMUP_RUNS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --skip-parity) SKIP_PARITY=1; shift ;;
    --skip-bench) SKIP_BENCH=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

IFS=',' read -r -a VAR_LIST <<< "${VARIANTS}"
if [[ ${#VAR_LIST[@]} -eq 0 ]]; then
  echo "--variants resolved to empty list" >&2
  exit 1
fi

normalize_variant() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | xargs
}

test_name_for_variant() {
  case "$1" in
    iq2-xxs) echo "cpu_vs_metal_parity_iq2_xxs" ;;
    iq2-xs) echo "cpu_vs_metal_parity_iq2_xs" ;;
    iq2-s) echo "cpu_vs_metal_parity_iq2_s" ;;
    iq3-s) echo "cpu_vs_metal_parity_iq3_s" ;;
    *) return 1 ;;
  esac
}

for i in "${!VAR_LIST[@]}"; do
  v="$(normalize_variant "${VAR_LIST[$i]}")"
  if ! test_name_for_variant "${v}" >/dev/null; then
    echo "invalid variant in --variants: ${v}" >&2
    exit 1
  fi
  VAR_LIST[$i]="${v}"
done

VARIANTS_NORM="$(IFS=','; echo "${VAR_LIST[*]}")"

if [[ "${SKIP_PARITY}" -eq 0 ]]; then
  echo "[iq-kernel] running parity tests..."
  for v in "${VAR_LIST[@]}"; do
    test_name="$(test_name_for_variant "${v}")"
    echo "[iq-kernel] parity ${v} (${test_name})"
    cargo test -q --features metal "${test_name}"
  done
fi

if [[ "${SKIP_BENCH}" -eq 0 ]]; then
  echo "[iq-kernel] running microbench..."
  cargo run --release --features metal --example iq2_matmul_bench -- \
    --variants "${VARIANTS_NORM}" \
    --m "${M}" \
    --out-dim "${OUT_DIM}" \
    --in-dim "${IN_DIM}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --runs "${RUNS}" \
    --dtype "${DTYPE}"
fi

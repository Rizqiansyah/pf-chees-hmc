#!/usr/bin/env bash
# R2-WP1 CPU smoke: 2 models x {re-seed on, re-seed off, no refresh}, C=64 D=10 W=50, seeds 1,2,3.
# Rows -> ~/pf-runs/r2-wp1-smoke/ (outside git); a copy of the rows is committed under
# results/r2/R2-WP1/smoke_rows/. Run from the checkout root.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
WT="$(pwd)"
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"
OUT="$HOME/pf-runs/r2-wp1-smoke"
mkdir -p "$OUT"
run() {  # model tag extra-flags...
  local model="$1" tag="$2"; shift 2
  R2WP1_SIDECAR="$OUT/${model}__${tag}.sidecar.jsonl" \
  pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
    python "$WT/results/r2/R2-WP1/smoke_driver.py" \
    --models "$model" --chains 64 --draws 10 --warmups 50 --seeds 1,2,3 --repeat 1 \
    --arm "r2wp1_${tag}" --wp R2-WP1 --orientation covariance --iterate ma \
    --out "$OUT/${model}__${tag}.jsonl" "$@" > "$OUT/${model}__${tag}.log" 2>&1
}
# 8schools: the three arms run in parallel. logit_1000x25 runs them one after another (CPU).
run 8schools reseed_on  --metric-refresh sample_diag --metric-refresh-reseed &
run 8schools reseed_off --metric-refresh sample_diag --no-metric-refresh-reseed &
run 8schools norefresh  --metric-refresh none &
wait
run logit_1000x25 reseed_on  --metric-refresh sample_diag --metric-refresh-reseed
run logit_1000x25 reseed_off --metric-refresh sample_diag --no-metric-refresh-reseed
run logit_1000x25 norefresh  --metric-refresh none
echo done

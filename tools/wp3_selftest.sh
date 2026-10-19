#!/bin/sh
# WP3 self-test: re-derive plan §2e through the public sampler and the bias harness.
#
# One process per (model, warmup, seed): device peak memory is process-cumulative and JAX 0.11.2 has
# no reset on this card, so a shared process could not attribute peak bytes to a cell. Rep 0 (cold)
# and rep 1 (warm) stay together inside the process, which is what makes them cold and warm.
#
# Every GPU process here goes through `flock /tmp/pf-gpu.lock` AND sets PF_GPU_LOCK=1, which is what
# --timed asserts on. A cell that could not take the lock waits rather than racing, and a cell that
# somehow runs unlocked dies in require_gpu_lock() instead of producing a void timing.
#
# Config notes: pf-draws 64 + init_pick iid is §2e's own workaround for the C=1024 Pathfinder-pool
# OOM (797 MiB) and the stratified-scoring OOM (500 MiB), which WP5 fixes. K (superchains) is off
# here so the numbers are comparable with §2e; nested R-hat with K=8 is exercised separately.
#
# NOT set -e: one failed cell must not truncate the rest of the matrix (it did -- a traceback in cell
# 1 of 17 ended the run 6 minutes in with no complaint). Failures are echoed and every cell writes its
# own row with the error field set, so nothing is lost either way.
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"
WT="${WT:-$HOME/pf/wp3}"
OUT="${OUT:-$WT/results/runs/WP3__selftest.jsonl}"
LEGACY="${LEGACY:-$HOME/pf/legacy}"
ARM="${ARM:-legacy}"
REPEAT="${REPEAT:-2}"
K="${K:-0}"
WS_8S="${WS_8S:-100,2000}"
WS_LOGIT="${WS_LOGIT:-100,200,400}"
SEEDS="${SEEDS:-1,2,3}"
mkdir -p "$(dirname "$OUT")"

cell() {  # $1 = extra r1_bias args, rest = the python args
  flock /tmp/pf-gpu.lock env PF_GPU_LOCK=1 \
    pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
      python "$WT/tools/r1_bias.py" "$@"
}

# 1) Pathfinder starts alone: §2e's b2_max = 2.71 (8 Schools) and 0.305 (logistic).
# STARTS_MODELS= (set but empty) skips this block; ${VAR:-default} would treat empty as unset and
# re-run the starts cells, which is exactly what a nested-R-hat re-run does not want to pay for.
STARTS_MODELS="${STARTS_MODELS-8schools logit_1000x25}"
for M in $STARTS_MODELS; do
  echo "=== starts $M"
  cell --models "$M" --chains 1024 --pf-draws 64 --init-pick iid --arm pf_starts \
       --legacy "$LEGACY" --repeat 1 --starts-only --warmups 0 --seeds 1 --out "$OUT" \
    || echo "!!! starts FAILED $M (continuing)"
done

# 2) The R1 grid: smallest W with b2_max < 0.01 on every seed, and gradients per chain there.
for SEED in $(echo "$SEEDS" | tr ',' ' '); do
  for W in $(echo "$WS_8S" | tr ',' ' '); do
    echo "=== cell 8schools W=$W seed=$SEED"
    cell --models 8schools --chains 1024 --draws 10 --warmups "$W" --seeds "$SEED" \
         --repeat "$REPEAT" --arm "$ARM" --legacy "$LEGACY" --pf-draws 64 --init-pick iid \
         --superchains "$K" --timed --out "$OUT" \
      || echo "!!! cell FAILED 8schools W=$W seed=$SEED (continuing)"
  done
  for W in $(echo "$WS_LOGIT" | tr ',' ' '); do
    echo "=== cell logit_1000x25 W=$W seed=$SEED"
    cell --models logit_1000x25 --chains 1024 --draws 10 --warmups "$W" --seeds "$SEED" \
         --repeat "$REPEAT" --arm "$ARM" --legacy "$LEGACY" --pf-draws 64 --init-pick iid \
         --superchains "$K" --timed --out "$OUT" \
      || echo "!!! cell FAILED logit W=$W seed=$SEED (continuing)"
  done
done
echo "=== done: $OUT"

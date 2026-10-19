#!/bin/sh
# WP3 self-test: re-derive plan §2e through the public sampler and the bias harness.
#
# One process per (model, warmup, seed): device peak memory is process-cumulative and JAX 0.11.2 has
# no reset on this card, so a shared process could not attribute peak bytes to a cell. Rep 0 (cold)
# and rep 1 (warm) stay together inside the process, which is what makes them cold and warm.
#
# Whole thing runs under one GPU lock (A5): flock is taken once here so the per-cell processes are
# serialised against every other task's work as well as each other.
#
# Config notes: pf-draws 64 + init_pick iid is §2e's own workaround for the C=1024 Pathfinder-pool
# OOM (797 MiB) and the stratified-scoring OOM (500 MiB), which WP5 fixes. K (superchains) is off
# here so the numbers are comparable with §2e; nested R-hat with K=8 is exercised separately.
set -eu
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"
WT="${WT:-$HOME/pf/wp3}"
OUT="${OUT:-$WT/results/runs/WP3__selftest.jsonl}"
LEGACY="${LEGACY:-$HOME/pf/legacy}"
ARM="${ARM:-legacy}"
REPEAT="${REPEAT:-2}"
K="${K:-0}"
EXTRA="${EXTRA:-}"
mkdir -p "$(dirname "$OUT")"

run_cell() {  # model warmup seed
  echo "=== cell model=$1 W=$2 seed=$3"
  flock /tmp/pf-gpu.lock env PF_GPU_LOCK=1 \
    pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
    python "$WT/tools/r1_bias.py" \
      --models "$1" --chains 1024 --draws 10 --warmups "$2" --seeds "$3" \
      --repeat "$REPEAT" --arm "$ARM" --legacy "$LEGACY" \
      --pf-draws 64 --init-pick iid --superchains "$K" --timed --out "$OUT" $EXTRA
}

# Pathfinder starts alone (§2e's b2_max = 2.71 on 8 Schools, 0.305 on logistic).
for M in ${MODELS:-8schools logit_1000x25}; do
  flock /tmp/pf-gpu.lock env PF_GPU_LOCK=1 \
    pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" \
    python "$WT/tools/r1_bias.py" --models "$M" --chains 1024 --pf-draws 64 --init-pick iid \
      --arm pf_starts --legacy "$LEGACY" --repeat 1 --starts-only \
      --warmups 0 --seeds 1 --out "$OUT" $EXTRA
done

for SEED in 1 2 3; do
  run_cell 8schools 100 "$SEED"
  run_cell 8schools 2000 "$SEED"
  run_cell logit_1000x25 100 "$SEED"
  run_cell logit_1000x25 200 "$SEED"
  run_cell logit_1000x25 400 "$SEED"
done
echo "=== done: $OUT"

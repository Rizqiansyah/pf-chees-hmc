#!/bin/bash
# F2k GPU part (lead): float32 against float64 on the A2000, at the merged head, in one session. Shapes as in
# F2f-F2i: C=1024, D=10, each model at its M2a smallest passing W. One process per job, under tools/gpu_run.sh.
# float32 is PyMC's own switch, PYTENSOR_FLAGS=floatX=float32, set before PyTensor is imported.
# Usage: results/f2/drivers/f2k_gpu.sh [MODEL ...]   (default: the four ladder models)
# Jobs per model (JSON -> results/f2/runs/F2k/gpu_<job>__<model>.json, logs -> ~/pf-runs/f2k/logs):
#   f64_same, f32_same   --calls 3, the same data, cache off (the tools' default); float64 first, then float32
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
OUT=results/f2/runs/F2k; LOG=~/pf-runs/f2k/logs
mkdir -p "$OUT" "$LOG"
TREE=$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')
echo "$(date -Is) f2k_gpu sha=$(git rev-parse --short HEAD) trees=$TREE" | tee -a "$LOG/driver.log"
declare -A W=([8schools]=25 [gev]=25 [logit_1000x25]=50 [radon_noncentered]=300)
declare -A FLAGS=([f64]="blas__ldflags=-llapack -lblas -lcblas -lm" [f32]="floatX=float32,blas__ldflags=-llapack -lblas -lcblas -lm")
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(8schools gev logit_1000x25 radon_noncentered)

for m in "${MODELS[@]}"; do
  for fx in f64 f32; do
    if [[ "$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')" != "$TREE" || -n "$(git status --porcelain -- pfcheeshmc tools)" ]]; then
      echo "$(date -Is) ABORT code changed ($m $fx)" | tee -a "$LOG/driver.log"; exit 3; fi
    job="${fx}_same"
    load=$(cut -d' ' -f1-3 /proc/loadavg)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')
    t0=$(date +%s)
    tools/gpu_run.sh timeout 3600 env -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR PYTENSOR_FLAGS="${FLAGS[$fx]}" \
        pixi run python tools/warm_stages.py "$MAIN" "$m" 1024 10 "${W[$m]}" --calls 3 --json "$OUT/gpu_${job}__${m}.json" \
        < /dev/null > "$LOG/${job}__${m}.log" 2>&1
    rc=$?
    echo "$(date -Is) job=$job model=$m rc=$rc wall=$(( $(date +%s)-t0 ))s load_before=[$load] gpu_before=[$gpu]" \
        | tee -a "$LOG/driver.log"
    grep -E "^(floatX|disk cache|call)" "$LOG/${job}__${m}.log" | sed 's/^/    /' | tee -a "$LOG/driver.log"
  done
done
ls -ld ~/.cache/pfcheeshmc 2>&1 | sed 's/^/    real default dir: /' | tee -a "$LOG/driver.log"
echo "$(date -Is) f2k_gpu done" | tee -a "$LOG/driver.log"

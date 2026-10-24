#!/bin/bash
# F2i GPU check (lead): the cheaper widening (_rebuild, one fold), which runs in every call's "rest"
# stage. Same shapes as F2f-F2h: C=1024, D=10, each model at its M2a smallest passing W. One process per job, under
# tools/gpu_run.sh.
# Usage: results/f2/drivers/f2i_gpu.sh [MODEL ...]   (default: 8schools radon_noncentered gev)
# Job per model (JSON -> results/f2/runs/F2i/<job>__<model>.json, logs -> ~/pf-runs/f2i/logs):
#   f2i_same  --calls 3, the same data, cache off (the tools' default): compare with F2h's f2h_same
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
OUT=results/f2/runs/F2i; LOG=~/pf-runs/f2i/logs
mkdir -p "$OUT" "$LOG"
TREE=$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')
echo "$(date -Is) f2i_gpu sha=$(git rev-parse --short HEAD) trees=$TREE" | tee -a "$LOG/driver.log"
declare -A W=([8schools]=25 [gev]=25 [logit_1000x25]=50 [radon_noncentered]=300)
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(8schools radon_noncentered gev)

for m in "${MODELS[@]}"; do
  if [[ "$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')" != "$TREE" || -n "$(git status --porcelain -- pfcheeshmc tools)" ]]; then
    echo "$(date -Is) ABORT code changed ($m)" | tee -a "$LOG/driver.log"; exit 3; fi
  load=$(cut -d' ' -f1-3 /proc/loadavg)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')
  t0=$(date +%s)
  tools/gpu_run.sh timeout 3600 env -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR \
      pixi run python tools/warm_stages.py "$MAIN" "$m" 1024 10 "${W[$m]}" --calls 3 --json "$OUT/f2i_same__${m}.json" \
      < /dev/null > "$LOG/f2i_same__${m}.log" 2>&1
  rc=$?
  echo "$(date -Is) job=f2i_same model=$m rc=$rc wall=$(( $(date +%s)-t0 ))s load_before=[$load] gpu_before=[$gpu]" \
      | tee -a "$LOG/driver.log"
  grep -E "^(disk cache|call)" "$LOG/f2i_same__${m}.log" | sed 's/^/    /' | tee -a "$LOG/driver.log"
done
ls -ld ~/.cache/pfcheeshmc 2>&1 | sed 's/^/    real default dir: /' | tee -a "$LOG/driver.log"
echo "$(date -Is) f2i_gpu done" | tee -a "$LOG/driver.log"

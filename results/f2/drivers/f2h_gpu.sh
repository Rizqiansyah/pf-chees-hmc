#!/bin/bash
# F2h GPU check (lead): the host gather (np.take + jax.device_put) that replaced jnp.broadcast_to for the
# pm_chains < chains rows. The default pm_chains = 1 goes through it. Same shapes as F2f/F2g: C=1024, D=10, each
# model at its M2a smallest passing W. One process per job, under tools/gpu_run.sh.
# Usage: results/f2/drivers/f2h_gpu.sh [MODEL ...]   (default: 8schools radon_noncentered)
# Job per model (JSON -> results/f2/runs/F2h/<job>__<model>.json, logs -> ~/pf-runs/f2h/logs):
#   f2h_same  --calls 3, the same data, cache off (the tools' default): compare with F2g's f2g_same
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
OUT=results/f2/runs/F2h; LOG=~/pf-runs/f2h/logs
mkdir -p "$OUT" "$LOG"
TREE=$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')
echo "$(date -Is) f2h_gpu sha=$(git rev-parse --short HEAD) trees=$TREE" | tee -a "$LOG/driver.log"
declare -A W=([8schools]=25 [gev]=25 [logit_1000x25]=50 [radon_noncentered]=300)
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(8schools radon_noncentered)

for m in "${MODELS[@]}"; do
  if [[ "$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')" != "$TREE" || -n "$(git status --porcelain -- pfcheeshmc tools)" ]]; then
    echo "$(date -Is) ABORT code changed ($m)" | tee -a "$LOG/driver.log"; exit 3; fi
  load=$(cut -d' ' -f1-3 /proc/loadavg)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')
  t0=$(date +%s)
  tools/gpu_run.sh timeout 3600 env -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR \
      pixi run python tools/warm_stages.py "$MAIN" "$m" 1024 10 "${W[$m]}" --calls 3 --json "$OUT/f2h_same__${m}.json" \
      < /dev/null > "$LOG/f2h_same__${m}.log" 2>&1
  rc=$?
  echo "$(date -Is) job=f2h_same model=$m rc=$rc wall=$(( $(date +%s)-t0 ))s load_before=[$load] gpu_before=[$gpu]" \
      | tee -a "$LOG/driver.log"
  grep -E "^(disk cache|call)" "$LOG/f2h_same__${m}.log" | sed 's/^/    /' | tee -a "$LOG/driver.log"
done
ls -ld ~/.cache/pfcheeshmc 2>&1 | sed 's/^/    real default dir: /' | tee -a "$LOG/driver.log"
echo "$(date -Is) f2h_gpu done" | tee -a "$LOG/driver.log"

#!/bin/bash
# F2f GPU stage timing (plan F2f): main (git archive 232ae58) vs f2 at the M4 shapes, C=1024, D=10, each
# model at its M2a smallest passing W. One process per job, each under tools/gpu_run.sh (the GPU flock).
# Usage: results/f2/drivers/f2f_gpu.sh [MODEL ...]   (default: the 4 ladder models)
# Jobs per model (JSON -> results/f2/runs/F2f/<job>__<model>.json, logs -> ~/pf-runs/f2f/logs):
#   main_same / f2_same      --calls 3, the same data every call ("repeat runs" in one process)
#   main_new  / f2_new       --calls 3 --new-data 0, new observed data every call ("new data every run")
#   f2_cache_fill            --calls 1 --cache-dir D --new-data 0 (fresh empty D: the cache cold, writes it)
#   f2_cache_new             --calls 2 --cache-dir D --new-data 10 (fresh process, NEW data, the cache warm)
#   main_cache_fill          --calls 1, JAX's three cache env vars, fresh empty D' (main has no helper)
#   main_cache_same          --calls 1, same env, same data (main's best case for repeat runs in new processes)
#   main_cache_new           --calls 1, same env, --new-data 10 (main with new data and a warm cache)
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
MAINTREE=~/pf-runs/f2f/main_tree           # git archive 232ae58 | tar -x
OUT=results/f2/runs/F2f; LOG=~/pf-runs/f2f/logs; CACHE=~/pf-runs/f2f/cache
mkdir -p "$OUT" "$LOG" "$CACHE"
TREE=$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')
echo "$(date -Is) f2f_gpu sha=$(git rev-parse --short HEAD) trees=$TREE main_tree=$MAINTREE" | tee -a "$LOG/driver.log"
declare -A W=([8schools]=25 [gev]=25 [logit_1000x25]=50 [radon_noncentered]=300)
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(8schools gev logit_1000x25 radon_noncentered)
CACHE_ENV="JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1"

job() {  # job NAME MODEL TREE ARGS... [-- ENV...]
  local name=$1 m=$2 tree=$3; shift 3
  local args=() envs=()
  while [[ $# -gt 0 && $1 != "--" ]]; do args+=("$1"); shift; done
  [[ ${1:-} == "--" ]] && { shift; envs=("$@"); }
  if [[ "$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')" != "$TREE" || -n "$(git status --porcelain -- pfcheeshmc tools)" ]]; then
    echo "$(date -Is) ABORT code changed ($name $m)" | tee -a "$LOG/driver.log"; exit 3; fi
  local load; load=$(cut -d' ' -f1-3 /proc/loadavg)
  local gpu; gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')
  local t0; t0=$(date +%s)
  tools/gpu_run.sh timeout 3600 env -u JAX_PLATFORM_NAME "${envs[@]}" pixi run python tools/warm_stages.py \
      "$tree" "$m" 1024 10 "${W[$m]}" "${args[@]}" --json "$OUT/${name}__${m}.json" \
      < /dev/null > "$LOG/${name}__${m}.log" 2>&1
  local rc=$?
  echo "$(date -Is) job=$name model=$m rc=$rc wall=$(( $(date +%s)-t0 ))s load_before=[$load] gpu_before=[$gpu] args=${args[*]} env=${envs[*]}" \
      | tee -a "$LOG/driver.log"
  grep -E "^call" "$LOG/${name}__${m}.log" | sed 's/^/    /' | tee -a "$LOG/driver.log"
}

for m in "${MODELS[@]}"; do
  rm -rf "$CACHE/f2_$m" "$CACHE/main_$m"
  job main_same "$m" "$MAINTREE" --calls 3
  job f2_same   "$m" "$MAIN"     --calls 3
  job main_new  "$m" "$MAINTREE" --calls 3 --new-data 0
  job f2_new    "$m" "$MAIN"     --calls 3 --new-data 0
  job f2_cache_fill "$m" "$MAIN" --calls 1 --cache-dir "$CACHE/f2_$m" --new-data 0
  job f2_cache_new  "$m" "$MAIN" --calls 2 --cache-dir "$CACHE/f2_$m" --new-data 10
  job main_cache_fill "$m" "$MAINTREE" --calls 1 -- JAX_COMPILATION_CACHE_DIR="$CACHE/main_$m" $CACHE_ENV
  job main_cache_same "$m" "$MAINTREE" --calls 1 -- JAX_COMPILATION_CACHE_DIR="$CACHE/main_$m" $CACHE_ENV
  job main_cache_new  "$m" "$MAINTREE" --calls 1 --new-data 10 -- JAX_COMPILATION_CACHE_DIR="$CACHE/main_$m" $CACHE_ENV
  du -sh "$CACHE/f2_$m" "$CACHE/main_$m" 2>/dev/null | tee -a "$LOG/driver.log"
done
echo "$(date -Is) f2f_gpu done" | tee -a "$LOG/driver.log"

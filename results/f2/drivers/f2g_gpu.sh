#!/bin/bash
# F2g GPU re-check (lead): the jitted start repair and the default on-disk cache, at the F2f shapes
# (C=1024, D=10, each model at its M2a smallest passing W). One process per job, under tools/gpu_run.sh.
# Usage: results/f2/drivers/f2g_gpu.sh [MODEL ...]   (default: the 4 ladder models)
# Jobs per model (JSON -> results/f2/runs/F2g/<job>__<model>.json, logs -> ~/pf-runs/f2g/logs):
#   f2g_same         --calls 3, the same data, cache off (the tools' default): compare with F2f's f2_same
#   f2g_default_fill --calls 1 --new-data 0, PFCHEESHMC_COMPILATION_CACHE=1 with a fresh empty XDG_CACHE_HOME
#                    (the library's default directory, written for the first time)
#   f2g_default_new  --calls 2 --new-data 10, the same XDG_CACHE_HOME in a fresh process: NEW data, the default
#                    cache warm. Compare with F2f's f2_cache_new (the explicit helper)
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
OUT=results/f2/runs/F2g; LOG=~/pf-runs/f2g/logs; XDG=~/pf-runs/f2g/xdg
mkdir -p "$OUT" "$LOG" "$XDG"
TREE=$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')
echo "$(date -Is) f2g_gpu sha=$(git rev-parse --short HEAD) trees=$TREE" | tee -a "$LOG/driver.log"
declare -A W=([8schools]=25 [gev]=25 [logit_1000x25]=50 [radon_noncentered]=300)
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(8schools gev logit_1000x25 radon_noncentered)

job() {  # job NAME MODEL ARGS... [-- ENV...]
  local name=$1 m=$2; shift 2
  local args=() envs=()
  while [[ $# -gt 0 && $1 != "--" ]]; do args+=("$1"); shift; done
  [[ ${1:-} == "--" ]] && { shift; envs=("$@"); }
  if [[ "$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')" != "$TREE" || -n "$(git status --porcelain -- pfcheeshmc tools)" ]]; then
    echo "$(date -Is) ABORT code changed ($name $m)" | tee -a "$LOG/driver.log"; exit 3; fi
  local load; load=$(cut -d' ' -f1-3 /proc/loadavg)
  local gpu; gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')
  local t0; t0=$(date +%s)
  tools/gpu_run.sh timeout 3600 env -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR "${envs[@]}" \
      pixi run python tools/warm_stages.py "$MAIN" "$m" 1024 10 "${W[$m]}" "${args[@]}" --json "$OUT/${name}__${m}.json" \
      < /dev/null > "$LOG/${name}__${m}.log" 2>&1
  local rc=$?
  echo "$(date -Is) job=$name model=$m rc=$rc wall=$(( $(date +%s)-t0 ))s load_before=[$load] gpu_before=[$gpu] args=${args[*]} env=${envs[*]}" \
      | tee -a "$LOG/driver.log"
  grep -E "^(disk cache|call)" "$LOG/${name}__${m}.log" | sed 's/^/    /' | tee -a "$LOG/driver.log"
}

for m in "${MODELS[@]}"; do
  rm -rf "$XDG/$m"; mkdir -p "$XDG/$m"
  job f2g_same "$m" --calls 3
  job f2g_default_fill "$m" --calls 1 --new-data 0 -- PFCHEESHMC_COMPILATION_CACHE=1 XDG_CACHE_HOME="$XDG/$m"
  job f2g_default_new  "$m" --calls 2 --new-data 10 -- PFCHEESHMC_COMPILATION_CACHE=1 XDG_CACHE_HOME="$XDG/$m"
  d="$XDG/$m/pfcheeshmc/jax"
  echo "    $d: $(du -sh "$d" | cut -f1), $(find "$d" -maxdepth 1 -type f | wc -l) files, modes $(stat -c %a "$XDG/$m/pfcheeshmc" "$d" | tr '\n' ' ')subdirs: $(find "$d" -mindepth 1 -type d -printf '%P ' )" \
      | tee -a "$LOG/driver.log"
done
ls -ld ~/.cache/pfcheeshmc 2>&1 | sed 's/^/    real default dir: /' | tee -a "$LOG/driver.log"
echo "$(date -Is) f2g_gpu done" | tee -a "$LOG/driver.log"

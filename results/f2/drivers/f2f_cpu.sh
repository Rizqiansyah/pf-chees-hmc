#!/bin/bash
# F2f CPU NUTS per-call walls (plan F2f "CPU NUTS first call"), never during a GPU job: 32 chains on 32 cores,
# each model at its M5 passing (tune, draws). One process per (model, PyTensor cache state):
#   cold  PYTENSOR_FLAGS compiledir = a new empty directory (no C module of anything compiled yet)
#   warm  the default compiledir (run 2's M5 compiled these models with their own data)
# Each process times: first call, repeat (same model object), rebuilt (same data), new_data x2
# (results/f2/drivers/cpu_nuts_calls.py). Data seeds are new per run, so "new data" is never a C-cache hit.
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
OUT=results/f2/runs/F2f; LOG=~/pf-runs/f2f/logs; mkdir -p "$OUT" "$LOG"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH="$MAIN"
BLAS="blas__ldflags=-llapack -lblas -lcblas -lm"
declare -A TD=([8schools]="200 250" [gev]="200 100" [logit_1000x25]="200 250" [radon_noncentered]="200 800")
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(8schools gev logit_1000x25 radon_noncentered)
gpu_wait() { while pgrep -f 'tools/(r1_bias|warm_stages).py' > /dev/null; do sleep 30; done; }
S=$(( $(date +%s) % 100000 * 10 ))
echo "$(date -Is) f2f_cpu sha=$(git rev-parse --short HEAD) data_seed_base=$S" | tee -a "$LOG/driver.log"
for m in "${MODELS[@]}"; do
  read -r tune draws <<< "${TD[$m]}"
  for state in cold warm; do
    gpu_wait
    S=$((S + 2))
    if [[ $state == cold ]]; then
      dir=~/pf-runs/f2f/ptc_cold_$m; rm -rf "$dir"; mkdir -p "$dir"; flags="compiledir=$dir,$BLAS"
    else
      flags="$BLAS"
    fi
    load=$(cut -d' ' -f1-3 /proc/loadavg); t0=$(date +%s)
    env -u JAX_PLATFORM_NAME PYTENSOR_FLAGS="$flags" pixi run python results/f2/drivers/cpu_nuts_calls.py \
        "$m" "$tune" "$draws" --data-seeds "$((S - 1)),$S" --json "$OUT/cpu_${state}__${m}.json" \
        < /dev/null > "$LOG/cpu_${state}__${m}.log" 2>&1
    echo "$(date -Is) cpu model=$m state=$state rc=$? wall=$(( $(date +%s)-t0 ))s load_before=[$load]" | tee -a "$LOG/driver.log"
    grep -E "^$m " "$LOG/cpu_${state}__${m}.log" | sed 's/^/    /' | tee -a "$LOG/driver.log"
  done
done
echo "$(date -Is) f2f_cpu done" | tee -a "$LOG/driver.log"

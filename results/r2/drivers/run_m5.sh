#!/bin/bash
# M5-cpu driver (plan Part B M5, A5): CPU NUTS time-to-pass via tools/t3_cpu_nuts.py, one process per
# (model, tune, draws), seeds 1,2,3. Never overlaps a GPU batch: before each process it waits until no
# r1_bias.py process exists. One untimed warm-up cell per model first, so PyTensor's one-off C
# compile is not in any timed row (R2-WP3a review note). Rows -> results/r2/runs/M5__cpu_nuts.jsonl.
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
LOG=~/pf-runs/r2-logs/M5; mkdir -p "$LOG"; OUT=results/r2/runs/M5__cpu_nuts.jsonl
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH="$MAIN"
TREE=$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')
echo "$(date -Is) step=M5 sha=$(git rev-parse --short HEAD) trees=$TREE" | tee -a "$LOG/driver.log"
gpu_wait() { while pgrep -f 'tools/r1_bias.py' > /dev/null; do sleep 30; done; }
for m in 8schools gev logit_1000x25 radon_noncentered; do
  gpu_wait
  pixi run python tools/t3_cpu_nuts.py --models "$m" --tune 200 --draws 100 --seeds 99 \
      --out ~/pf-runs/r2-logs/M5/warmup_untimed.jsonl < /dev/null > "$LOG/warmup__$m.log" 2>&1
  echo "$(date -Is) warmup model=$m rc=$?" | tee -a "$LOG/driver.log"
done
for m in 8schools gev logit_1000x25 radon_noncentered; do for t in 200 500 1000; do for d in 50 100 250 400; do
  gpu_wait
  if [[ "$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')" != "$TREE" ]]; then
    echo "$(date -Is) ABORT code changed during M5" | tee -a "$LOG/driver.log"; exit 3; fi
  t0=$(date +%s)
  pixi run python tools/t3_cpu_nuts.py --models "$m" --tune "$t" --draws "$d" --seeds 1,2,3 --out "$OUT" \
      < /dev/null > "$LOG/${m}_t${t}_d${d}.log" 2>&1
  echo "$(date -Is) model=$m tune=$t draws=$d rc=$? wall=$(( $(date +%s)-t0 ))s rows=$(wc -l < "$OUT")" | tee -a "$LOG/driver.log"
done; done; done
echo "$(date -Is) step=M5 done" | tee -a "$LOG/driver.log"

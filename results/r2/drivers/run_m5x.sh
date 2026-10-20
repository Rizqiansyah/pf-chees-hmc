#!/bin/bash
# M5x (DEVIATED, extension): radon has no CPU-NUTS pass in the plan grid (draws <= 400): every cell is
# "not resolvable" (floor > 0.005) with b2 0.001-0.003. Extend draws for radon only so R2-T4 has a CPU
# time-to-pass. Same driver contract as run_m5.sh; runs only when no GPU job is alive.
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
LOG=~/pf-runs/r2-logs/M5; OUT=results/r2/runs/M5__cpu_nuts.jsonl
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH="$MAIN"
for t in 200 500 1000; do for d in 800 1600; do
  while pgrep -f '[t]ools/r1_bias.py' > /dev/null || pgrep -f '[g]3_meanshift.py' > /dev/null; do sleep 30; done
  t0=$(date +%s)
  pixi run python tools/t3_cpu_nuts.py --models radon_noncentered --tune "$t" --draws "$d" --seeds 1,2,3 --out "$OUT" \
      < /dev/null > "$LOG/radon_noncentered_t${t}_d${d}.log" 2>&1
  echo "$(date -Is) M5x model=radon_noncentered tune=$t draws=$d rc=$? wall=$(( $(date +%s)-t0 ))s rows=$(wc -l < "$OUT")" | tee -a "$LOG/driver.log"
done; done
echo "$(date -Is) step=M5x done" | tee -a "$LOG/driver.log"

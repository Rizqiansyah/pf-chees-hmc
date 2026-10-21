#!/bin/bash
# M3 G3 step: GEV posterior-mean shift (sd units and z) for the three M3 arms, seeds 1-5 (= M3 seeds).
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
mkdir -p ~/pf-runs/r2-logs/M3g
export MODEL=gev SEEDS=1,2,3,4,5 CHAINS=32 DRAWS=1250 WARMUP=2000 OUT=results/r2/runs/M3g__gev_meanshift.jsonl
export ARMS_JSON='[["m3:ma", {"iterate": "ma"}], ["m3:raw", {"iterate": "raw"}], ["m3:raw_refresh", {"iterate": "raw", "metric_refresh": "sample_diag", "metric_refresh_warmup": 1000}]]'
tools/gpu_run.sh timeout 7200 env PYTHONPATH="$MAIN" pixi run python results/r2/drivers/g3_meanshift.py \
    < /dev/null > ~/pf-runs/r2-logs/M3g/run.log 2>&1
echo "$(date -Is) step=M3g rc=$? rows=$(wc -l < "$OUT")" | tee -a ~/pf-runs/r2-logs/M3g/driver.log

#!/bin/bash
# Run-2 measurement driver (plan A3/A5). Usage: run_step.sh STEP CMDFILE [TIMEOUT_S]
# CMDFILE: one job per line, "<rowfile-name> <r1_bias.py args...>" ('#' lines skipped).
# Each job is one process under the GPU lock with the timeout INSIDE the lock, from $MAIN with
# PYTHONPATH=$MAIN, rows to results/r2/runs/<STEP>__<rowfile-name>.jsonl. Code freeze: every job
# asserts the pfcheeshmc/ and tools/ trees (committed and working) still match those at step start;
# results/ commits (LEDGER, STATUS) are allowed during a step.
set -u
MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; cd "$MAIN" || exit 2
STEP="$1"; CMDFILE="$2"; TO="${3:-10800}"
LOG=~/pf-runs/r2-logs/"$STEP"; mkdir -p "$LOG" results/r2/runs
SHA=$(git rev-parse --short HEAD)
TREE=$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')
echo "$(date -Is) step=$STEP sha=$SHA trees=$TREE cmdfile=$CMDFILE" | tee -a "$LOG/driver.log"
n=0
while read -r name args; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  n=$((n+1))
  if [[ -n "$(git status --porcelain -- pfcheeshmc tools)" \
        || "$(git rev-parse HEAD:pfcheeshmc HEAD:tools | tr '\n' ' ')" != "$TREE" ]]; then
    echo "$(date -Is) ABORT code changed during step $STEP (job $n $name)" | tee -a "$LOG/driver.log"; exit 3
  fi
  out="results/r2/runs/${STEP}__${name}.jsonl"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  tools/gpu_run.sh timeout "$TO" env PYTHONPATH="$MAIN" pixi run python tools/r1_bias.py $args \
      --timed --out "$out" < /dev/null > "$LOG/job$(printf %02d $n)__${name}.log" 2>&1
  rc=$?
  echo "$(date -Is) job=$n name=$name rc=$rc wall=$(( $(date +%s)-t0 ))s rows=$(wc -l < "$out" 2>/dev/null || echo 0) args=$args" \
      | tee -a "$LOG/driver.log"
done < "$CMDFILE"
echo "$(date -Is) step=$STEP done jobs=$n" | tee -a "$LOG/driver.log"

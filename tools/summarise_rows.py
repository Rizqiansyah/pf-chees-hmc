"""Summarise pf-r1/v1 rows: median per (arm, model, cold) with the counts the gates need.

Usage: pixi run python tools/summarise_rows.py results/runs/WP1__rebaseline.jsonl [model ...]

Reads only warm rows unless --cold is passed: A5 says decisions use warm rows and the two must never
be mixed. `n` is the number of seeds in the group, and a group with n < 5 is not a decision.
"""
import collections
import json
import statistics as st
import sys


def main(path, want_cold=False, models=()):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    voided = [r for r in rows if str(r.get("error") or "").startswith("VOID")]
    rows = [r for r in rows if not str(r.get("error") or "").startswith("VOID")]
    if voided:
        print(f"[audit] {len(voided)} VOID row(s) excluded: "
              + ", ".join(f"{r['arm']}/{r['model']}/seed{r['seed']}/rep{r['rep']}" for r in voided))
    seen = {}
    for r in rows:
        k = (r["wp"], r["arm"], r["model"], r["seed"], r["rep"])
        seen[k] = seen.get(k, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    print(f"[audit] duplicate (wp,arm,model,seed,rep) keys: {len(dups)}"
          + ("" if not dups else " -> " + ", ".join(str(k) for k in dups)))
    if models:
        rows = [r for r in rows if r["model"] in models]
    groups = collections.defaultdict(list)
    for r in rows:
        if bool(r["cold"]) != want_cold:
            continue
        groups[(r["arm"], r["model"])].append(r)

    cols = [("wall_s", "wall"), ("frozen_eps", "eps"), ("frozen_L_nominal", "L"),
            ("steps_mean", "steps"), ("grads_per_chain_warmup", "g/chain_wu"),
            ("grads_per_ess_worst", "g/ESS"), ("ess_s_worst", "ESS/s"),
            ("min_bulk_ess", "minBulkESS"), ("min_tail_ess", "minTailESS"),
            ("divs_per_1k", "divs/1k"), ("warmup_divs", "wuDivs"), ("ebfmi_min", "EBFMI")]
    head = f"{'arm':16s} {'model':18s} {'n':>2s} {'err':>3s} {'maxRhat':>8s} {'peakMB':>7s} "
    head += " ".join(f"{lbl:>11s}" for _, lbl in cols)
    print(("COLD" if want_cold else "WARM") + " rows  " + path)
    print(head)
    for (arm, model), rs in sorted(groups.items()):
        med = lambda key: (round(st.median([r[key] for r in rs if r.get(key) is not None]), 4)
                           if any(r.get(key) is not None for r in rs) else None)
        rhat = max([r["max_rhat"] for r in rs if r.get("max_rhat") is not None] or [float("nan")])
        errs = sum(1 for r in rs if r.get("error"))
        peak = max([r["peak_mem_bytes"] or 0 for r in rs] or [0]) / 1e6
        line = f"{arm:16s} {model:18s} {len(rs):2d} {errs:3d} {rhat:8.4f} {peak:7.0f} "
        line += " ".join(f"{str(med(k)):>11s}" for k, _ in cols)
        print(line)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "results/runs/WP1__rebaseline.jsonl"
    main(path, want_cold="--cold" in sys.argv, models=tuple(args[1:]))

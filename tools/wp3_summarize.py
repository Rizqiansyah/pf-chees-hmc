#!/usr/bin/env python
"""Turn pf-r1/v1 rows into the WP3 deliverable: per (model, arm), the smallest warmup that passes
G-R1 on EVERY seed, and the gradients per chain there.

Plan §6 defines the gate that way on purpose — a W that passes on 2 of 3 seeds is not a pass — so the
aggregation is code, not something hand-copied into a table.

b2_max is per (seed, rep); only warm rows (rep >= 1) decide, per A5. A cell is eligible only if
C*D >= 2,000 and its noise floor is <= 0.005; cells that fail those are reported as "not resolvable"
rather than as passes or failures, because the plan forbids reading them either way.

Usage: python tools/wp3_summarize.py results/runs/WP3__selftest.jsonl
"""
import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_cell(rows, warm_only=True):
    """{(model, arm, W, seed): row} — last row wins, so a re-run replaces a tagged/contended one."""
    out = {}
    for r in rows:
        if r["warmup"] in (None, 0) or r["b2_max"] is None:
            continue                      # pf_starts rows and error rows are not warmup cells
        if r.get("pooled") is True or r.get("seed") == "pooled":
            continue                      # pf-r2/v1 G-R1-pool row: a diagnostic, not a seed
        if warm_only and r["rep"] == 0:
            continue
        out[(r["model"], r["arm"], r["warmup"], r["seed"])] = r
    return out


def summarise(rows, warm_only=True, floor_limit=0.005, b2_limit=0.01):
    cells = per_cell(rows, warm_only)
    by_model = defaultdict(lambda: defaultdict(list))
    for (model, arm, W, seed), r in sorted(cells.items()):
        by_model[model][arm].append((W, seed, r))

    table = []
    for model, arms in sorted(by_model.items()):
        for arm, entries in sorted(arms.items()):
            seeds = sorted({s for _, s, _ in entries})
            by_w = defaultdict(dict)
            for W, seed, r in entries:
                by_w[W][seed] = r
            rows_out = []
            smallest = None
            for W in sorted(by_w):
                got = by_w[W]
                state, detail = "no data", ""
                if len(got) < len(seeds):
                    state = f"incomplete ({len(got)}/{len(seeds)} seeds)"
                else:
                    b2s = [got[s]["b2_max"] for s in seeds]
                    fl = [got[s]["b2_floor"] for s in seeds]
                    cd = got[seeds[0]]["chains"] * got[seeds[0]]["draws_per_chain"]
                    if cd < 2000:
                        state = "not eligible (C*D < 2000)"
                    elif max(fl) > floor_limit:
                        state = f"not resolvable (floor {max(fl):.4f})"
                    elif max(b2s) < b2_limit:
                        state = "PASS"
                        if smallest is None:
                            smallest = (W, [got[s]["grads_per_chain_warmup"] for s in seeds],
                                        [got[s]["grads_per_chain_sampling"] for s in seeds])
                    else:
                        state = f"fail (b2_max {max(b2s):.4f})"
                rows_out.append((W, state, max((v["b2_max"] for v in got.values()), default=None),
                                 max((v["b2_floor"] for v in got.values()), default=None)))
            table.append({"model": model, "arm": arm, "seeds": seeds, "cells": rows_out,
                          "smallest_passing_W": smallest[0] if smallest else None,
                          "grads_per_chain_at_that_W": None if not smallest else
                          [sum(x) for x in zip(smallest[1], smallest[2])],
                          "warm_only": warm_only})
    return table


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/runs/WP3__selftest.jsonl"
    rows = load(path)
    print(f"{len(rows)} rows from {path}")
    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"  {len(errors)} error rows: "
              + "; ".join(f"{r['model']}/W{r['warmup']}/s{r['seed']}: {r['error'][:70]}"
                          for r in errors[:5]))
    for t in summarise(rows):
        print(f"\n{t['model']}  arm={t['arm']}  seeds={t['seeds']}")
        for W, state, b2, floor in t["cells"]:
            print(f"   W={W:>5}  b2_max={'' if b2 is None else format(b2, '.5f')}  "
                  f"floor={'' if floor is None else format(floor, '.5f')}  {state}")
        if t["smallest_passing_W"] is None:
            print("   -> no W in this set passes G-R1 on every seed")
        else:
            print(f"   -> smallest W passing G-R1 on every seed: {t['smallest_passing_W']}, "
                  f"gradients per chain (warmup+sampling) = {t['grads_per_chain_at_that_W']}")


if __name__ == "__main__":
    main()

"""Summarise pf-r1/v1 and pf-r2/v1 rows.

Default mode: median per (arm, model, cold) with the counts the gates need.
Usage: pixi run python tools/summarise_rows.py results/runs/WP1__rebaseline.jsonl [model ...]

Reads only warm rows unless --cold is passed: A5 says decisions use warm rows and the two must never
be mixed. `n` is the number of seeds in the group, and a group with n < 5 is not a decision.

`smallest-pass` mode (R2-WP3a): per model, the cheapest cell that passes G-R1 on EVERY seed.
Usage: pixi run python tools/summarise_rows.py smallest-pass 'results/r2/runs/T1__*.jsonl' --arm A
       pixi run python tools/summarise_rows.py smallest-pass results/r2/runs/M5__cpu_nuts.jsonl
Grid and cost: R1 rows are (warmup W, cost W); M5 CPU-NUTS rows (`step == "M5"`) are
(tune, draws, cost tune+draws). The verdict is recomputed with `r1_bias.g_r1` from b2_max, b2_floor
and chains*draws, so every row kind is judged by the one gate. Pooled rows (`pooled: true`,
`seed: "pooled"`, G-R1-pool) are a diagnostic: they are printed beside the answer and never enter
the every-seed test.
"""
import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


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


# ---------------------------------------------------------------- smallest-pass

def _g_r1(r):
    """The gate, recomputed. Lazy import: r1_bias pulls in jax, which the plain table does not need."""
    sys.path.insert(0, HERE)
    import r1_bias

    draws = r.get("draws_per_chain") if r.get("draws_per_chain") is not None else r.get("draws")
    return r1_bias.g_r1(r["b2_max"], r["b2_floor"], r["chains"] * draws)


def _is_pooled(r):
    return r.get("pooled") is True or r.get("seed") == "pooled"


def _is_m5(r):
    return r.get("step") == "M5" or (r.get("tune") is not None and r.get("warmup") is None)


def _grads(r):
    if r.get("grads_per_chain_total") is not None:
        return r["grads_per_chain_total"]
    if r.get("grads_per_chain_warmup") is not None and r.get("grads_per_chain_sampling") is not None:
        return r["grads_per_chain_warmup"] + r["grads_per_chain_sampling"]
    return None


def _med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def _expand(patterns):
    paths = []
    for p in patterns:
        hits = sorted(glob.glob(os.path.expanduser(p))) or ([p] if os.path.exists(p) else [])
        paths += [h for h in hits if h not in paths]
    if not paths:
        raise FileNotFoundError(f"no row file matches {list(patterns)}")
    return paths


def _scoreable(rows, warm_only):
    """Rows that can enter a verdict; the exclusions are counted, never silent."""
    keep, audit = [], collections.Counter()
    for r in rows:
        if str(r.get("error") or "").startswith("VOID"):
            audit["VOID"] += 1
        elif r.get("error"):
            audit["error"] += 1
        elif r.get("b2_max") is None or r.get("b2_floor") is None:
            audit["no b2"] += 1
        elif warm_only and r.get("cold"):
            audit["cold"] += 1
        else:
            keep.append(r)
    return keep, audit


def _cells(rows, grid_of, group_of):
    """{group: {grid point: {seed: row}}} and {group: {grid point: pooled row}}; last row wins."""
    cells = collections.defaultdict(lambda: collections.defaultdict(dict))
    pooled = collections.defaultdict(dict)
    dup = 0
    for r in rows:
        g, pt = group_of(r), grid_of(r)
        if pt is None:
            continue                                        # pf_starts rows have no warmup
        if _is_pooled(r):
            pooled[g][pt] = r
            continue
        dup += r["seed"] in cells[g][pt]
        cells[g][pt][r["seed"]] = r
    return cells, pooled, dup


def _state(got, seeds):
    """(state, verdicts) for one grid point: PASS only if EVERY expected seed has a `pass`."""
    verdicts = {s: _g_r1(got[s]) for s in seeds if s in got}
    if len(verdicts) < len(seeds):
        return f"incomplete ({len(verdicts)}/{len(seeds)} seeds)", verdicts
    bad = sorted({v for v in verdicts.values() if v != "pass"})
    if not bad:
        return "PASS", verdicts
    order = ("fail", "not resolvable", "not eligible")      # worst first
    worst = min(bad, key=lambda v: next((i for i, o in enumerate(order) if v.startswith(o)), 9))
    return worst.split(" (")[0], verdicts


def smallest_pass(rows, mode="auto", arm=None, models=None, seeds=None, max_w=None,
                  warm_only=False):
    """Per (model, arm): the cheapest grid point that passes G-R1 on every seed.

    mode "r1": grid = warmup W, cost = W.  mode "m5": grid = (tune, draws), cost = tune + draws,
    ties broken by median wall_s then tune.  Returns (results, audit).
    """
    rows = [r for r in rows if (arm is None or r.get("arm") == arm)
            and (not models or r.get("model") in models)]
    if mode == "auto":
        usable = [r for r in rows if r.get("warmup") is not None or r.get("tune") is not None]
        mode = "m5" if usable and all(_is_m5(r) for r in usable) else "r1"
    rows, audit = _scoreable(rows, warm_only)
    if mode == "m5":
        grid_of = lambda r: (int(r["tune"]), int(r["draws"])) if r.get("tune") is not None else None
        cost = lambda pt: pt[0] + pt[1]
    else:
        grid_of = lambda r: r["warmup"] if r.get("warmup") not in (None, 0) else None
        cost = lambda pt: pt
    cells, pooled, dup = _cells(rows, grid_of, lambda r: (r["model"], r.get("arm")))
    audit["cells with >1 row (last wins)"] = dup

    out = []
    for g in sorted(cells, key=lambda k: (str(k[0]), str(k[1]))):
        pts = sorted(cells[g], key=lambda p: (cost(p), p))
        if mode == "r1" and max_w is not None:
            pts = [p for p in pts if p <= max_w]
        want = sorted(seeds) if seeds else sorted({s for p in pts for s in cells[g][p]})
        table, winners = [], []
        for p in pts:
            state, verdicts = _state(cells[g][p], want)
            got = [cells[g][p][s] for s in want if s in cells[g][p]]
            table.append({"pt": p, "state": state, "verdicts": verdicts,
                          "wall_s": _med([r.get("wall_s") for r in got])})
            if state == "PASS":
                winners.append(p)
        win = None
        if winners:
            win = min(winners, key=lambda p: (cost(p), _med([cells[g][p][s].get("wall_s")
                                                             for s in want]) or float("inf"), p)) \
                if mode == "m5" else winners[0]
        res = {"model": g[0], "arm": g[1], "mode": mode, "seeds": want, "cells": table,
               "max_cost": max((cost(p) for p in pts), default=None), "max_pt": pts[-1] if pts else None,
               "pt": win}
        if win is not None:
            got = cells[g][win]
            res.update(
                grads=_med([_grads(got[s]) for s in want]),
                wall_s=_med([got[s].get("wall_s") for s in want]),
                core_seconds=_med([got[s].get("core_seconds") for s in want]),
                per_seed=[(s, got[s]["b2_max"], got[s]["b2_floor"]) for s in want],
                pooled=(None if win not in pooled[g]
                        else (pooled[g][win]["b2_max"], pooled[g][win]["b2_floor"])))
        out.append(res)
    for r in rows:                                          # stored verdict vs the gate, as audit
        stored = r.get("g_r1")
        if stored is not None and not _is_pooled(r) and stored != _g_r1(r):
            audit["stored g_r1 != recomputed"] += 1
    return out, audit


def _bf(x):
    return "-" if x is None else f"{x:.4g}"


def format_smallest_pass(results, audit, cells=False):
    lines = [f"[audit] {k}: {v}" for k, v in sorted(audit.items()) if v]
    for res in results:
        m5 = res["mode"] == "m5"
        head = f"{res['model']:18s} arm={res['arm']}  seeds={res['seeds']}"
        if res["pt"] is None:
            tail = (f"none <= max cost {res['max_cost']} (tune+draws, grid up to tune={res['max_pt'][0]}"
                    f" draws={res['max_pt'][1]})" if m5 and res["max_pt"] else
                    f"none <= max W {res['max_pt']}")
            lines.append(f"{head}\n   -> {tail}")
        else:
            seeds = "  ".join(f"s{s}:{_bf(b)}/{_bf(f)}" for s, b, f in res["per_seed"])
            what = (f"tune={res['pt'][0]} draws={res['pt'][1]} (cost {sum(res['pt'])})" if m5
                    else f"W={res['pt']}")
            cost = (f"wall={_bf(res['wall_s'])}s core_s={_bf(res['core_seconds'])} " if m5 else "")
            pooled = "pooled b2/floor: -" if res["pooled"] is None \
                else f"pooled b2/floor: {_bf(res['pooled'][0])}/{_bf(res['pooled'][1])} (diagnostic)"
            lines.append(f"{head}\n   -> smallest pass {what}: {cost}median grads/chain="
                         f"{_bf(res['grads'])}\n      per-seed b2/floor  {seeds}\n      {pooled}")
        if cells:
            for c in res["cells"]:
                v = " ".join(f"s{s}={x}" for s, x in sorted(c["verdicts"].items()))
                lines.append(f"      {str(c['pt']):>12s}  {c['state']:24s} {v}")
    return "\n".join(lines)


def smallest_pass_main(argv):
    ap = argparse.ArgumentParser(prog="summarise_rows.py smallest-pass")
    ap.add_argument("paths", nargs="+", help="row file(s) or glob(s)")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--models", default="", help="comma list")
    ap.add_argument("--seeds", default="", help="expected seeds (default: those seen per model/arm)")
    ap.add_argument("--max-w", type=int, default=None, help="r1 mode: ignore W above this")
    ap.add_argument("--mode", default="auto", choices=["auto", "r1", "m5"])
    ap.add_argument("--warm-only", action="store_true", help="drop rows with cold: true")
    ap.add_argument("--cells", action="store_true", help="also print every grid point's verdicts")
    a = ap.parse_args(argv)
    paths = _expand(a.paths)
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    print(f"smallest-pass  {len(rows)} rows from {len(paths)} file(s): {', '.join(paths[:3])}"
          + (" ..." if len(paths) > 3 else ""))
    results, audit = smallest_pass(
        rows, mode=a.mode, arm=a.arm, models=tuple(m for m in a.models.split(",") if m),
        seeds=[int(s) for s in a.seeds.split(",") if s], max_w=a.max_w, warm_only=a.warm_only)
    print(format_smallest_pass(results, audit, cells=a.cells))
    return 0 if results else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["smallest-pass"]:
        sys.exit(smallest_pass_main(sys.argv[2:]))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "results/runs/WP1__rebaseline.jsonl"
    main(path, want_cold="--cold" in sys.argv, models=tuple(args[1:]))

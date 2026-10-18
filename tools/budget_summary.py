"""Equal-total-draw-budget comparison: wall seconds to produce the SAME number of samples.

Grouped by (model, total draws, chains). GPU shows cold (first cell at that chain count, carries
the JAX trace) and warm, at the shipped 2,000-step warmup; the warmup-1,000 variant is printed
separately because it is a different adaptation budget. CPU is one pass with tune=1,000.

Quality columns decide whether the samples were worth buying: max R-hat and the bulk ESS of the
worst single variable (= ess_s_worstvar x wall), which is what you actually spend wall time to get.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "pf-runs" / "budget.jsonl")
rows = [json.loads(l) for l in PATH.read_text().splitlines() if l.startswith("{")]

data = defaultdict(lambda: defaultdict(list))          # (model,total,chains) -> backend -> rows
for r in rows:
    if "error" in r:
        continue
    data[(r["model"], r["draws_total"], r["chains"])][r["backend"]].append(r)


def pick(rows_, want_warmup=None):
    """Best row for a backend: preferred warmup, then warm over cold, then fastest."""
    pool = [r for r in rows_ if (want_warmup is None or r["warmup"] == want_warmup)] or rows_
    warm = [r for r in pool if not r["cold"]]
    return (warm or pool)[0] if (warm or pool) else {}


def worst_ess(r):
    return r.get("ess_s_worstvar", 0) * r.get("wall_s", 0) if r else float("nan")


errors = [r for r in rows if "error" in r]
if errors:
    seen = set()
    for r in errors:
        k = (r["model"], r["backend"], r.get("chains"))
        if k not in seen:
            seen.add(k)
            print(f"ERROR cell (x{sum(1 for e in errors if (e['model'], e['backend'], e.get('chains')) == k)}):"
                  f" {r['model']} {r['backend']} chains={r.get('chains')} {r['error'][:70]}")

for model in sorted({k[0] for k in data}):
    for budget in sorted({k[1] for k in data if k[0] == model}):
        print(f"\n=== {model}  —  {budget:,} post-warmup draws total")
        print(f"{'chains':>7} {'GPU cold':>9} {'GPU warm':>9} {'CPU':>8} {'GPU/CPU':>8} "
              f"{'R-hat G':>8} {'R-hat C':>8} {'ESS1 G':>8} {'ESS1 C':>8} {'div/1k G':>9} {'div/1k C':>9}")
        for ch in sorted({k[2] for k in data if k[0] == model and k[1] == budget}):
            rec = data[(model, budget, ch)]
            gpu_rows, cpu_rows = rec.get("gpu", []), rec.get("cpu_nuts", [])
            gw = pick(gpu_rows, 2000)
            gc = pick([r for r in gpu_rows if r["cold"]], 2000)
            cpu = pick(cpu_rows)
            ratio = f"{gw['wall_s'] / cpu['wall_s']:.2f}x" if gw and cpu else "-"
            fmt = lambda r, k, p=4: (f"{r[k]:>{p}}" if r else " " * (p - 1) + "-")
            print(f"{ch:>7} {fmt(gc, 'wall_s', 9):>9} {fmt(gw, 'wall_s', 9):>9} {fmt(cpu, 'wall_s', 8):>8} {ratio:>8} "
                  f"{fmt(gw, 'max_rhat')} {fmt(cpu, 'max_rhat')} {worst_ess(gw):>8.0f} {worst_ess(cpu):>8.0f} "
                  f"{fmt(gw, 'divs_per_1k_draws', 9)} {fmt(cpu, 'divs_per_1k_draws', 9)}")

print("\n=== GPU at warmup 1,000 (adaptation-budget variant, seed 43)")
print(f"{'model':<18} {'total':>7} {'chains':>7} {'wall':>7} {'R-hat':>8} {'ESS1':>8}")
for (model, budget, ch), rec in sorted(data.items()):
    r = pick(rec.get("gpu", []), 1000)
    if r:
        print(f"{model:<18} {budget:>7} {ch:>7} {r['wall_s']:>7.1f} {r['max_rhat']:>8.4f} {worst_ess(r):>8.0f}")

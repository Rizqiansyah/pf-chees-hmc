"""F2g GPU re-check: tables from results/f2/runs/F2g/*.json against F2f's (tools/warm_stages.py output). Markdown."""
import json, os, statistics
R = os.path.join(os.path.dirname(__file__), "..", "runs")
MODELS = ["8schools", "gev", "logit_1000x25", "radon_noncentered"]
STAGES = ["init", "pathfinder", "eps0", "warmup", "sampling", "transform", "rest"]
def load(step, job, m):
    p = os.path.join(R, step, f"{job}__{m}.json")
    return json.load(open(p)) if os.path.exists(p) else None
def warm(rows):
    return statistics.mean(r["wall"] for r in rows if r["call"] > 1)
def warm_init(rows):
    return statistics.mean(r["stage"]["init"][0] for r in rows if r["call"] > 1)
print("Same data, cache off: F2f (3e0a739) -> F2g. Call 1 wall / XLA compiles; init of call 1 (s / compiles); "
      "warm wall and warm init, mean of calls 2-3")
print()
print("| model | call 1, F2f | call 1, F2g | init call 1, F2f | init call 1, F2g | warm, F2f | warm, F2g | warm init, F2f | warm init, F2g |")
print("|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    a, b = load("F2f", "f2_same", m), load("F2g", "f2g_same", m)
    if not (a and b):
        continue
    print(f"| {m} | {a[0]['wall']:.1f} s / {a[0]['compiles']} | {b[0]['wall']:.1f} s / {b[0]['compiles']} | "
          f"{a[0]['stage']['init'][0]:.2f} / {a[0]['stage']['init'][1]} | {b[0]['stage']['init'][0]:.2f} / {b[0]['stage']['init'][1]} | "
          f"{warm(a):.2f} s | {warm(b):.2f} s | {warm_init(a):.2f} | {warm_init(b):.2f} |")
print()
print("First call in a fresh process with NEW data and a warm disk cache: F2f explicit helper (`--cache-dir`) vs F2g "
      "library default (`PFCHEESHMC_COMPILATION_CACHE=1`, fresh XDG_CACHE_HOME). Wall (XLA compile s); fill = the "
      "process that wrote it")
print()
print("| model | F2f fill | F2f warm, new data | F2g default fill | F2g default warm, new data | F2g init, warm cache |")
print("|---|---|---|---|---|---|")
for m in MODELS:
    g = [load("F2f", "f2_cache_fill", m), load("F2f", "f2_cache_new", m),
         load("F2g", "f2g_default_fill", m), load("F2g", "f2g_default_new", m)]
    if not all(g):
        continue
    print(f"| {m} | " + " | ".join(f"{x[0]['wall']:.1f} s ({x[0]['compile_s']:.1f})" for x in g)
          + f" | {g[3][0]['stage']['init'][0]:.2f} s |")

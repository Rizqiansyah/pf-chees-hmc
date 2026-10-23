"""F2f: tables from results/f2/runs/F2f/*.json (tools/warm_stages.py output). Prints markdown."""
import glob, json, os, statistics, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "runs", "F2f")
MODELS = ["8schools", "gev", "logit_1000x25", "radon_noncentered"]
STAGES = ["init", "pathfinder", "eps0", "warmup", "sampling", "transform", "rest"]
def load(job, m):
    p = os.path.join(D, f"{job}__{m}.json")
    return json.load(open(p)) if os.path.exists(p) else None
def warm(rows):
    w = [r for r in rows if r["call"] > 1]
    return statistics.mean(r["wall"] for r in w), {s: statistics.mean(r["stage"][s][0] for r in w) for s in STAGES}
print("| model | main cold | main warm | main warm, new data | f2 cold | f2 warm | f2 warm, new data | warm speed-up |")
print("|---|---|---|---|---|---|---|---|")
for m in MODELS:
    ms, mn, fs, fn = (load(j, m) for j in ("main_same", "main_new", "f2_same", "f2_new"))
    mw, fw = warm(ms)[0], warm(fs)[0]
    print(f"| {m} | {ms[0]['wall']:.1f} s | {mw:.2f} s | {warm(mn)[0]:.2f} s | {fs[0]['wall']:.1f} s | {fw:.2f} s | "
          f"{warm(fn)[0]:.2f} s | {mw / fw:.1f}x |")
print()
print("First call in a fresh process (s): no disk cache; disk cache warm with the same data; disk cache warm with NEW data")
print()
print("| model | main, no cache | main, cache, same data | main, cache, new data | f2, no cache | f2, cache, new data |")
print("|---|---|---|---|---|---|")
for m in MODELS:
    g = {j: load(j, m) for j in ("main_same", "main_cache_same", "main_cache_new", "f2_new", "f2_cache_new")}
    print(f"| {m} | " + " | ".join(f"{g[j][0]['wall']:.1f} ({g[j][0]['compile_s']:.1f} compile)" for j in g) + " |")
print()
print("Warm call per stage, mean of calls 2-3, seconds (same data)")
print()
print("| model | tree | wall | " + " | ".join(STAGES) + " |")
print("|---|---|---|" + "---|" * len(STAGES))
for m in MODELS:
    for tree, job in (("main", "main_same"), ("f2", "f2_same")):
        w, st = warm(load(job, m))
        print(f"| {m} | {tree} | {w:.2f} | " + " | ".join(f"{st[s]:.2f}" for s in STAGES) + " |")
print()
print("Cold call per stage (call 1, same data), seconds / XLA compiles")
print()
print("| model | tree | wall | " + " | ".join(STAGES) + " |")
print("|---|---|---|" + "---|" * len(STAGES))
for m in MODELS:
    for tree, job in (("main", "main_same"), ("f2", "f2_same")):
        r = load(job, m)[0]
        print(f"| {m} | {tree} | {r['wall']:.1f} | " + " | ".join(f"{r['stage'][s][0]:.2f} / {r['stage'][s][1]}" for s in STAGES) + " |")

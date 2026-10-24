"""F2k GPU tables (markdown).
1. Warm stages, float64 vs float32, same session and head: results/f2/runs/F2k/gpu_{f64,f32}_same__<model>.json.
2. The R1 gate under float32 (results/f2/runs/F2k__float32.jsonl) against F2f's float64 gate rows
   (results/f2/runs/F2f__default.jsonl), same model, W, seeds 1-3."""
import json, os, statistics
R = os.path.join(os.path.dirname(__file__), "..", "runs")
MODELS = ["8schools", "gev", "logit_1000x25", "radon_noncentered"]
def load(fx, m):
    p = os.path.join(R, "F2k", f"gpu_{fx}_same__{m}.json")
    return json.load(open(p)) if os.path.exists(p) else None
warm = lambda rows, f: statistics.mean(f(r) for r in rows if r["call"] > 1)
print("Warm stages on the GPU, C=1024, D=10, same data, cache off. Call 1 wall / XLA compiles (compile s); warm = mean "
      "of calls 2-3")
print()
print("| model | floatX | call 1 | warm wall | warm warmup | warm sampling | warm pathfinder | warm init | compiles, calls 2/3 |")
print("|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for fx in ("f64", "f32"):
        r = load(fx, m)
        if not r:
            continue
        st = lambda k: warm(r, lambda c: c["stage"][k][0])
        print(f"| {m} | {r[0].get('floatX', fx)} | {r[0]['wall']:.1f} s / {r[0]['compiles']} ({r[0]['compile_s']:.1f}) | "
              f"{warm(r, lambda c: c['wall']):.2f} s | {st('warmup'):.2f} | {st('sampling'):.2f} | {st('pathfinder'):.2f} | "
              f"{st('init'):.3f} | {'/'.join(str(c['compiles']) for c in r[1:])} |")
print()
def rows(p):
    p = os.path.join(R, p)
    return [r for r in map(json.loads, open(p)) if not r.get("pooled")] if os.path.exists(p) else []
old, new = rows("F2f__default.jsonl"), rows("F2k__float32.jsonl")
print("R1 gate on the GPU: F2f float64 rows vs F2k float32 rows (seeds 1-3, rep 0; warm wall = median of rep 1)")
print()
print("| model | W | floatX | b2_max seeds 1-3 (floor) | G-R1 | eps | L | divs/1k | max R-hat | warm wall | cold wall |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for tag, rs in (("f64 (F2f)", old), ("f32", new)):
        r0 = sorted((r for r in rs if r["model"] == m and r["rep"] == 0), key=lambda r: r["seed"])
        r1 = [r for r in rs if r["model"] == m and r["rep"] == 1]
        if not r0:
            continue
        b2 = " ".join(f"{r['b2_max']:.4f}" for r in r0)
        fl = max(r["b2_floor"] for r in r0)
        ok = f"{sum(r['b2_max'] < 0.01 for r in r0)}/{len(r0)}"
        f = lambda k, fmt: " ".join(format(r[k], fmt) for r in r0)
        ww = f"{statistics.median(r['wall_s'] for r in r1):.2f} s" if r1 else "-"
        cold = [r for r in r0 if r["seed"] == 1][0]["wall_s"]
        print(f"| {m} | {r0[0]['warmup']} | {tag} | {b2} (≤{fl:.4f}) | {ok} | {f('frozen_eps', '.3g')} | "
              f"{f('frozen_L_nominal', '.0f')} | {f('divs_per_1k', '.2f')} | {f('max_rhat', '.2f')} | {ww} | {cold:.1f} s |")

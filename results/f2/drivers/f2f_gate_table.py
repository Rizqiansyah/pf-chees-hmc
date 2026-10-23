"""F2f: the R1 gate on f2 (results/f2/runs/F2f__default.jsonl) against run 2's M4 default rows
(results/r2/runs/M4__default.jsonl), same model, W, seeds 1-3. Prints markdown."""
import json, statistics
def rows(p):
    return [r for r in map(json.loads, open(p)) if not r.get("pooled")]
new, old = rows("results/f2/runs/F2f__default.jsonl"), rows("results/r2/runs/M4__default.jsonl")
G = lambda b2, fl: "pass" if b2 < 0.01 else "FAIL"
print("| model | W | tree | b2_max seeds 1-3 (floor) | G-R1 | eps | L | grads/chain | divs/1k | warm wall (median rep 1) | cold wall |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in ["8schools", "gev", "logit_1000x25", "radon_noncentered"]:
    for tag, rs in (("main (run 2 M4)", old), ("f2", new)):
        r0 = sorted((r for r in rs if r["model"] == m and r["rep"] == 0), key=lambda r: r["seed"])
        r1 = [r for r in rs if r["model"] == m and r["rep"] == 1]
        b2 = " ".join(f"{r['b2_max']:.4f}" for r in r0)
        fl = max(r["b2_floor"] for r in r0)
        ok = "pass 3/3" if all(r["b2_max"] < 0.01 for r in r0) else f"{sum(r['b2_max'] < 0.01 for r in r0)}/3"
        eps = " ".join(f"{r['frozen_eps']:.3f}" for r in r0)
        L = " ".join(f"{r['frozen_L_nominal']:.0f}" for r in r0)
        gr = " ".join(f"{r['grads_per_chain_total']:.0f}" for r in r0)
        dv = " ".join(f"{r['divs_per_1k']:.2f}" for r in r0)
        ww = statistics.median(r["wall_s"] for r in r1)
        cold = [r for r in r0 if r["seed"] == 1][0]["wall_s"]
        print(f"| {m} | {r0[0]['warmup']} | {tag} | {b2} (≤{fl:.4f}) | {ok} | {eps} | {L} | {gr} | {dv} | {ww:.2f} s | {cold:.1f} s |")

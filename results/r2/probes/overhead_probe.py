"""Where does a warm R1 call spend its fixed time? Compile log + cProfile on a warm call."""
import cProfile, io, logging, os, pstats, re, sys, time

MAIN = "/home/akbar/Documents/Python packages/pf-chees-hmc"
sys.path.insert(0, os.path.join(MAIN, "tools"))
import r1_bias  # noqa: F401  (sets PYTENSOR_FLAGS like the harness)
import bench_ladder as bl
import jax
import pfcheeshmc

name, W, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model = bl.model_specs()[name]()


def call(seed):
    t = time.time()
    idata = pfcheeshmc.sample_pf_chees_hmc(model, draws=10, chains=C, random_seed=seed,
                                           progressbar=False, chees_kwargs={"draws": W})
    jax.block_until_ready(jax.numpy.zeros(1))
    return time.time() - t, idata


w1, _ = call(1)
w2, _ = call(2)
print(f"[probe] {name} C={C} W={W}: call1 {w1:.2f}s  call2 {w2:.2f}s", flush=True)

recs = []


class H(logging.Handler):
    def emit(self, rec):
        recs.append(rec.getMessage())


jax.config.update("jax_log_compiles", True)
for lg in ("jax", "jax._src", "jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.pjit"):
    logging.getLogger(lg).addHandler(H())
    logging.getLogger(lg).setLevel(logging.DEBUG)

pr = cProfile.Profile()
pr.enable()
w3, idata = call(3)
pr.disable()
jax.config.update("jax_log_compiles", False)

tot = {"tracing": 0.0, "mlir": 0.0, "xla": 0.0}
n = {"tracing": 0, "mlir": 0, "xla": 0}
names = {}
for m in recs:
    sec = re.search(r"in ([0-9.eE+-]+) sec", m)
    if not sec:
        continue
    s = float(sec.group(1))
    key = "tracing" if "tracing" in m else "mlir" if "MLIR" in m else "xla" if "XLA compilation" in m else None
    if key:
        tot[key] += s
        n[key] += 1
        if key == "xla":
            fn = re.search(r"compilation of (\S+)", m)
            names[fn.group(1) if fn else "?"] = names.get(fn.group(1) if fn else "?", 0.0) + s
print(f"[probe] call3 (warm, profiled) {w3:.2f}s")
print(f"[probe] compiles during warm call: " + ", ".join(f"{k} n={n[k]} sum={tot[k]:.2f}s" for k in tot))
print("[probe] top XLA compiles:", sorted(((round(v, 2), k) for k, v in names.items()), reverse=True)[:12])
print("[probe] stage_s attr:", {k: v for k, v in idata.attrs.items() if "stage" in k or "time" in k.lower()})
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
print(s.getvalue()[:9000])

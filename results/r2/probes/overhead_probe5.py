"""Warm-call recompiles: same model object, seeds 1,1,2,3; count XLA compiles per call."""
import logging, os, re, sys, time
MAIN = "/home/akbar/Documents/Python packages/pf-chees-hmc"
sys.path.insert(0, os.path.join(MAIN, "tools"))
import r1_bias as rb
import jax, pfcheeshmc
name, W, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model = rb.build_model(name)
msgs = []
class H(logging.Handler):
    def emit(self, rec): msgs.append(rec.getMessage())
jax.config.update("jax_log_compiles", True)
h = H()
for lg in ("jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.pjit", "jax._src.compiler", "jax"):
    logging.getLogger(lg).addHandler(h); logging.getLogger(lg).setLevel(logging.DEBUG)
logging.getLogger("jax").propagate = False
CD = os.environ.get("PROBE_CACHE")
if CD:
    jax.config.update("jax_compilation_cache_dir", CD); jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0); jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
print("persistent cache dir:", jax.config.jax_compilation_cache_dir, flush=True)
for seed in (1, 7):
    msgs.clear(); t = time.time()
    pfcheeshmc.sample_pf_chees_hmc(model, draws=10, chains=C, random_seed=seed, progressbar=False,
                                   chees_kwargs={"draws": W})
    w = time.time() - t
    xla = [m for m in msgs if "XLA compilation" in m]
    hits = [m for m in msgs if "cache hit" in m.lower() or "Persistent compilation cache hit" in m]
    sec = sum(float(re.search(r"in ([0-9.eE+-]+) sec", m).group(1)) for m in xla if re.search(r"in ([0-9.eE+-]+) sec", m))
    print(f"[probe2] seed={seed} wall={w:.2f}s  xla_compiles={len(xla)} ({sec:.1f}s logged)  cache_hits={len(hits)}", flush=True)
    if seed == 7:
        for m in msgs:
            if "cache" in m.lower() and ("miss" in m.lower() or "writing" in m.lower() or "wrote" in m.lower()):
                print("    CACHE:", m[:160])
    print("   ", sorted({re.search(r'compilation of (\S+)', m).group(1) for m in xla if re.search(r'compilation of (\S+)', m)})[:12], flush=True)

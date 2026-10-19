"""WP1 re-baseline: the R2 matrix with the metric fixed (Part B WP1, "Re-baseline (GPU, R2 shape)").

One process = one (model, imm_source, arm). Seeds and REPEAT loop inside, so rep 0 of the first seed
carries the compile and rep 1 is warm; per A5 decisions use warm rows only. One process per cell
group because `jax.devices()[0].memory_stats()` has no reset in JAX 0.11.2 (verified), so the device
peak is cumulative for the process -- a monolithic sweep would report one number for everything.

Usage (always under the GPU lock, one model at a time):
  MAIN="/home/akbar/Documents/Python packages/pf-chees-hmc"; WT=~/pf/wp1
  MODEL=8schools IMM=mc ARM=wp1 tools/gpu_run.sh pixi run --manifest-path "$MAIN/pixi.toml" \
      env PYTHONPATH="$WT" python "$WT/tools/wp1_rebaseline.py"
  # legacy arm: point PYTHONPATH at the detached ecd9ba1 tree
  MODEL=8schools IMM=mc ARM=legacy ... env PYTHONPATH=$HOME/pf/legacy python "$WT/tools/wp1_rebaseline.py"

TRAP, cost a failed cell: never call `idata.posterior.to_array()` on a model with repeated-length
dims. radon NC has alpha_raw[85] and beta_raw[85] as *different* dims, and to_array() merges the
85-length and scalar dims into one index, so it materialises a cross-product: measured MemoryError,
137 TiB, shape (9, 32, 1250, 85, 85, 85, 85). Score per variable instead (bench_ladder.score already
iterates data_vars; the removed line was a jax.block_until_ready on to_array()). Each stage wrapper
already blocks on its own output, so nothing is lost. WP3's instrument must not repeat it.

Env: MODEL (one key of bench_ladder.model_specs), IMM (mc|lbfgs_lowrank|fisher_lowrank),
ARM (label only), SEEDS (comma list, default 43,44,45,46,47), REPEAT (default 2),
CHAINS (32), DRAWS (1250), WARMUP (2000), OUT (jsonl path).

Both arms are measured the same way: the stage timers and the warmup-gradient count are read by
wrapping PfCheesHMC's own methods from here, because the ecd9ba1 tree cannot carry new attributes.
"""
import json
import os
import sys
import time

WT = os.environ.get("WT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "tools"))

import jax
import numpy as np

import _common as C
import bench_ladder as bl
import pfcheeshmc
import pfcheeshmc.sampler as sm
from pfcheeshmc import sample_pf_chees_hmc

MODEL = os.environ["MODEL"]
IMM = os.environ.get("IMM", "mc")
ARM = os.environ.get("ARM", "wp1")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "43,44,45,46,47").split(",")]
REPEAT = int(os.environ.get("REPEAT", "2"))
CHAINS = int(os.environ.get("CHAINS", "32"))
DRAWS = int(os.environ.get("DRAWS", "1250"))
WARMUP = int(os.environ.get("WARMUP", "2000"))
# IMM_ORIENT is only added to pathfinder_kwargs when set, so an unset env keeps the tree default and
# the rows of the arms already in flight stay comparable.
IMM_ORIENT = os.environ.get("IMM_ORIENT") or ""
OUT = os.environ.get("OUT") or os.path.join(WT, "results", "runs", "WP1__rebaseline.jsonl")

# ---------------------------------------------------------------- stage instrumentation
STAGES = {                       # A8 stage_s key -> PfCheesHMC method
    "init": ("_pymc_adapt_diag_start", "_ensure_finite_inits"),
    "pathfinder": ("_run_pathfinder",),
    "eps0": ("_run_step_size_tune",),
    "warmup": ("_run_chees",),
    "sampling": ("_run_dynamic_hmc",),
}
TIMES: dict = {}
CAPTURED: dict = {}


def _wrap(method, key):
    orig = getattr(sm.PfCheesHMC, method)

    def inner(self, *a, **k):
        t0 = time.perf_counter()
        out = orig(self, *a, **k)
        first = out[0] if isinstance(out, tuple) else out
        try:
            jax.block_until_ready(first)
        except Exception:
            pass
        TIMES[key] = TIMES.get(key, 0.0) + (time.perf_counter() - t0)
        if method == "_run_chees":
            # (states, parameters, info): the info carries num_integration_steps per warmup step,
            # which is the only exact source of warmup gradient counts on the legacy tree.
            CAPTURED["params"] = out[1]
            CAPTURED["info"] = out[2]
        return out

    setattr(sm.PfCheesHMC, method, inner)


for key, methods in STAGES.items():
    for m in methods:
        _wrap(m, key)


def frozen_nominal_L(params, warmup_steps):
    """ceil(T/eps) from the kernel's own step-count closure (same rule as sampler._frozen_nominal_L).

    Duplicated here on purpose: the legacy arm must be measured with one method, and ecd9ba1 has no
    such helper. Kept in sync by test_metric_orientation.py.
    """
    base = max(int(warmup_steps) + 1000, 1024)
    best = 0.0
    for bits in (base.bit_length() - 1, base.bit_length(), base.bit_length() + 1):
        if bits < 1:
            continue
        try:
            best = max(best, float(params["integration_steps_fn"](jnp_int(2 ** bits - 2))))
        except Exception:
            continue
    return best


def jnp_int(v):
    return jax.numpy.asarray(v, dtype=jax.numpy.int64)


def main():
    C.require_gpu_lock(f"WP1 re-baseline {ARM}/{MODEL}/{IMM}")
    C.banner(model=MODEL, wp="WP1", arm=f"{ARM}:{IMM}" + (f":{IMM_ORIENT}" if IMM_ORIENT else ""))
    print(f"[cfg] tree={sm.__file__} MODEL={MODEL} IMM={IMM} ARM={ARM} C={CHAINS} "
          f"D={DRAWS} W={WARMUP} seeds={SEEDS} repeat={REPEAT} out={OUT}", flush=True)

    model = bl.model_specs()[MODEL]()
    # bench_ladder.load_notebook_model replaces pfcheeshmc.sample_pf_chees_hmc with a _tiny_idata
    # stub while parsing the notebook. A stubbed cell would time a fake InferenceData: fast wall,
    # zero divergences, no diagnostics, no exception. Pin the identity and record it in the row.
    assert pfcheeshmc.sample_pf_chees_hmc is bl.REAL_SAMPLE_PF, (
        "entry point was replaced by the notebook stub: "
        f"{pfcheeshmc.sample_pf_chees_hmc.__name__}")
    entry_point = pfcheeshmc.sample_pf_chees_hmc.__name__
    ip = model.initial_point()
    d = int(sum(np.size(v) for v in ip.values()))

    for seed in SEEDS:
        for rep in range(REPEAT):
            cold = rep == 0
            arm_label = f"{ARM}:{IMM}" + (f":{IMM_ORIENT}" if IMM_ORIENT else "")
            row = C.new_row(wp="WP1", arm=arm_label, model=MODEL, d=d, chains=CHAINS,
                            draws_per_chain=DRAWS, warmup=WARMUP, seed=seed, rep=rep, cold=cold,
                            kwargs={"pathfinder_kwargs": dict({"imm_source": IMM},
                                                              **({"imm_orientation": IMM_ORIENT}
                                                                 if IMM_ORIENT else {})),
                                    "chees_kwargs": {"draws": WARMUP},
                                    "progressbar": False, "entry_point": entry_point},
                            cpu_contended=False)
            TIMES.clear()
            CAPTURED.clear()
            t0 = time.perf_counter()
            try:
                assert pfcheeshmc.sample_pf_chees_hmc is bl.REAL_SAMPLE_PF
                idata = sample_pf_chees_hmc(
                    model, draws=DRAWS, chains=CHAINS, random_seed=seed, progressbar=False,
                    pathfinder_kwargs=(dict({"imm_source": IMM},
                                            **({"imm_orientation": IMM_ORIENT} if IMM_ORIENT else {}))),
                    chees_kwargs={"draws": WARMUP},
                )
                # Do NOT block on idata.posterior.to_array(): radon NC's posterior carries several
                # independent 85-sized dims, and to_array() materialises their cross product
                # (measured: MemoryError, 137 TiB for shape (9, 32, 1250, 85, 85, 85, 85)). Each stage
                # wrapper already calls block_until_ready on its own output, so the wall clock is
                # already gated on real device work.
                wall = time.perf_counter() - t0
            except Exception as exc:
                row["wall_s"] = round(time.perf_counter() - t0, 2)
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
                C.set_peak_mem(row)
                row["gpu_other_procs"] = C.gpu_other_procs()
                C.write_row(OUT, row)
                raise

            scored = bl.score(idata, wall, "gpu", MODEL, CHAINS, cold, draws=DRAWS, warmup=WARMUP)
            staged = {k: round(TIMES.get(k, 0.0), 3) for k in STAGES}
            accounted = sum(staged.values())
            row["wall_s"] = round(wall, 2)
            row["stage_s"] = C.stages(**staged, postproc=round(max(wall - accounted, 0.0), 3))
            row["frozen_eps"] = float(CAPTURED["params"]["step_size"])
            row["frozen_L_nominal"] = frozen_nominal_L(
                CAPTURED["params"], int(CAPTURED["params"].get("warmup_steps_used", WARMUP)))
            warm_n = np.asarray(CAPTURED["info"].num_integration_steps, dtype=np.int64)  # (steps,C)
            samp_n = np.asarray(idata.sample_stats["num_integration_steps"].values, dtype=np.int64)
            row["grads_per_chain_warmup"] = int(warm_n.sum() // warm_n.shape[1])
            row["grads_per_chain_sampling"] = int(samp_n.sum() // samp_n.shape[1])
            row["warmup_divs"] = int(np.asarray(CAPTURED["info"].is_divergent).sum())
            row["steps_mean"] = scored.get("steps_mean")
            row["divs_per_1k"] = scored.get("divs_per_1k_draws")
            row["max_rhat"] = scored.get("max_rhat")
            row["min_tail_ess"] = scored.get("min_tail_ess")
            row["ess_s_worst"] = scored.get("ess_s_worstvar")
            row["ebfmi_min"] = scored.get("ebfmi")
            min_bulk = float(scored["ess_s_worstvar"]) * wall
            row["min_bulk_ess"] = round(min_bulk, 0)
            total_grads = CHAINS * (row["grads_per_chain_warmup"] + row["grads_per_chain_sampling"])
            row["grads_per_ess_worst"] = round(total_grads / min_bulk, 3) if min_bulk > 0 else None
            row["error"] = None
            C.set_peak_mem(row)                      # AFTER the cell: peak is per-process (A8)
            row["gpu_other_procs"] = C.gpu_other_procs()
            C.write_row(OUT, row)


if __name__ == "__main__":
    main()

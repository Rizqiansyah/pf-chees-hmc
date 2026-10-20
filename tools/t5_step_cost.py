#!/usr/bin/env python
"""T5 (part): what the warmup's *optional* outputs cost per step.

Two of the four components of warmup cost are ablatable without building a parallel kernel, and only
these two are ever a choice the user makes:

  history   = chees_warmup_with_progress(stack_positions=True) - default   # the per-step position
                                                                                    # stack (WP5 removed it
                                                                                    # from the default path)
  callback  = chees_warmup_with_progress(progressbar=True)     - default   # the jax.debug.callback path

The other two (integration vs adaptation) come from paired `tools/r1_bias.py` rows instead of a
hand-built kernel: the sampling stage is the production kernel at frozen parameters, so
`kernel ~= sampling_s / sampling_grads x total_grads` and `adaptation = warmup_s - kernel`. Building
a standalone BlackJAX HMC scan was tried and rejected -- it has to reproduce the metric, momentum and
divergence semantics of the real path or the attribution is meaningless.

Boundary: everything here is inside the warmup scan. Pathfinder / eps0 / sampling / post-processing
are already timed per stage by r1_bias.

One JSON line to stdout and (with --out) to a row file. Run under tools/gpu_run.sh.
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import r1_bias as rb  # noqa: E402  (build_model restores bench_ladder's sampling stubs)
import _common as common  # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC  # noqa: E402
from pfcheeshmc.utils import chees_warmup_with_progress  # noqa: E402


def timed(fn, reps, warm=1):
    for _ in range(warm):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    common.require_gpu_lock("t5 warmup-output cost split")

    model = rb.build_model(a.model)
    s = PfCheesHMC(model, seed=a.seed)
    pos, imm = s._run_pathfinder(chains=a.chains, draws=4000, pf_pool_cap=4 * a.chains,
                                 imm_orientation="covariance")

    def chees(progressbar=False, stack_positions=False):
        return chees_warmup_with_progress(
            s.ld_fn, jax.random.key(a.seed), pos, 0.1, optax.adam(learning_rate=0.05),
            a.warmup, imm, target_acceptance_rate=0.651, max_leapfrog_steps=1000,
            progressbar=progressbar, iterate="raw", stack_positions=stack_positions)

    (st, pr), info = chees()
    hinfo = getattr(info, "info", info)
    steps = jnp.asarray(hinfo.num_integration_steps, dtype="int64")
    if steps.ndim == 1:
        steps = steps[:, None]
    total_steps = int(steps[:, 0].sum())

    t_default = timed(chees, a.reps)
    t_stack = timed(lambda: chees(stack_positions=True), a.reps)
    t_bar = timed(lambda: chees(progressbar=True), a.reps)

    out = {
        "model": a.model, "chains": a.chains, "warmup": a.warmup, "reps": a.reps,
        "git_sha": common.git_sha(), "branch": common.branch(), "device": str(jax.devices()[0]),
        "peak_mem_bytes": common.peak_mem_bytes(),
        "frozen_eps": round(float(jnp.asarray(pr["step_size"]).reshape(-1)[0]), 6),
        "total_leapfrog_per_chain": total_steps,
        "t_warmup_s": round(t_default, 2),
        "t_with_stack_s": round(t_stack, 2),
        "t_with_bar_s": round(t_bar, 2),
        "t_history_s": round(t_stack - t_default, 2),
        "t_callback_s": round(t_bar - t_default, 2),
        "history_pct": round(100.0 * (t_stack - t_default) / t_default, 1),
        "callback_pct": round(100.0 * (t_bar - t_default) / t_default, 1),
        "history_s_per_step": round((t_stack - t_default) / a.warmup, 6),
        "callback_s_per_step": round((t_bar - t_default) / a.warmup, 6),
    }
    print(json.dumps(out, sort_keys=True), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(out, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

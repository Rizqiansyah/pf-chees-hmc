"""WP5 deliverable 1: locate the device memory of the init path, stage by stage.

One process per configuration (required: on this card `peak_bytes_in_use` is
process-cumulative with no reset, so a second configuration in the same process
would only see the running maximum, not its own stages). Each stage appends one
pf-r1/v1 row; the stage name and both memory readings live in `kwargs`.

The stage sequence is exactly `PfCheesHMC.sample`'s own sequence, driven through
the real methods so the probe cannot drift from the code under test:

    pymc_init            pm.init_nuts (C chains, adapt_diag, tune)
    finite_inits         vmapped finite logp+grad mask + per-chain repair
    best_chain_score     the per-chain Python loop that seeds Pathfinder
    pathfinder_solve     L-BFGS fit from the best start (C-independent)
    pf_pool_draw         bj.vi.pathfinder.sample, C x pf_draws draws
    stratified_scoring   vmapped log-density over the scored subset
    covariance           (pool, d) flatten + X.T @ X
    eps0                 32-probe step-size search
    warmup_scan          the ChEES warmup (C x W)
    sampling_scan        the dynamic-HMC sampling (C x D)

An OOM run dies at the first failing stage; the last completed stage row plus the
stage banner printed before each stage name the wall.
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import blackjax as bj

from _common import new_row, write_row, banner, require_gpu_lock, gpu_other_procs


def device():
    for d in jax.devices():
        if d.platform == "gpu":
            return d
    return jax.devices()[0]


def mem_readings(dev):
    # JAX 0.11.2: per-device .memory_stats(); the peak key is process-cumulative with no
    # reset on this card (see tools/_common.peak_mem_bytes), which is why one process per config.
    s = dev.memory_stats()
    return int(s["bytes_in_use"]), int(s["peak_bytes_in_use"])


def run_cfg(model_name, C, D, W, pf_draws, seed, out_path):
    import tools.bench_ladder as bl
    from pfcheeshmc.sampler import PfCheesHMC

    model = bl.model_specs()[model_name]()
    dev = device()
    row0 = new_row(wp="WP5", arm="memprobe", model=model_name, chains=C,
                   draws_per_chain=D, warmup=W, seed=seed, rep=0, cold=True,
                   kwargs=json.dumps({"pf_draws": pf_draws}, sort_keys=True))
    d_flat = sum(v.size for v in jax.tree_util.tree_leaves(
        {rv.name: jnp.zeros_like(model.initial_point()[rv.name])
         for rv in model.value_vars}))
    row0["d"] = int(d_flat)
    banner(model=model_name, cold=True, wp="WP5", arm="memprobe")
    require_gpu_lock("WP5 memory probe")

    sampler = PfCheesHMC(model, seed=seed)
    key = jax.random.key(seed)
    key_fix, key_pf, key_samp, key_pick, key_choose = jax.random.split(key, 5)

    stages_done = []

    def mark(name, **extra):
        bi, pk = mem_readings(dev)
        stages_done.append(name)
        row = dict(row0)
        row["kwargs"] = json.dumps({**json.loads(row["kwargs"]), "stage": name,
                                    "bytes_in_use": bi, "peak_bytes_in_use": pk,
                                    **extra}, sort_keys=True)
        row["gpu_other_procs"] = gpu_other_procs()
        write_row(out_path, row)
        print(f"[mem] {name:20s} live={bi / 2**20:10.1f} MiB  peak={pk / 2**20:10.1f} MiB",
              flush=True)

    mark("start")

    t = time.perf_counter()
    init_stack = sampler._pymc_adapt_diag_start(C, init="adapt_diag", tune=500,
                                                random_seed=seed)
    mark("pymc_init", stage_s=round(time.perf_counter() - t, 3))
    t = time.perf_counter()
    init_stack = sampler._ensure_finite_inits(init_stack, key_fix)
    mark("finite_inits", stage_s=round(time.perf_counter() - t, 3))

    t = time.perf_counter()
    def _score_chain(c):
        x = {n: init_stack[n][c] for n in sampler.rvs}
        return float(sampler.ld_fn(x))
    scores = [(_score_chain(c), c) for c in range(C)]
    scores = [(s if jnp.isfinite(jnp.asarray(s)) else -jnp.inf, c) for s, c in scores]
    c_best = max(scores, key=lambda tup: tup[0])[1]
    x0_pf = {n: init_stack[n][c_best] for n in sampler.rvs}
    mark("best_chain_score", stage_s=round(time.perf_counter() - t, 3),
         c_best=int(c_best))
    t = time.perf_counter()
    pf_state, _ = bj.vi.pathfinder.approximate(
        rng_key=key_pf, initial_position=tuple(x0_pf[n] for n in sampler.rvs),
        logdensity_fn=sampler.ld_fn)
    mark("pathfinder_solve", stage_s=round(time.perf_counter() - t, 3))

    t = time.perf_counter()
    total = C * pf_draws
    samples_flat, _ = bj.vi.pathfinder.sample(key_samp, pf_state, total)
    mark("pf_pool_draw", stage_s=round(time.perf_counter() - t, 3), pool=int(total))
    t = time.perf_counter()
    stride = max(total // max(C * 64, 1), 1)
    subset_idx = jnp.arange(0, total, stride)
    lp_sub = jax.vmap(lambda i: sampler.ld_fn(
        jax.tree_util.tree_map(lambda x: x[i], samples_flat)))(subset_idx)
    from pfcheeshmc.sampler import _stratified_pool_idx
    pick_global = _stratified_pool_idx(lp_sub, subset_idx, C)
    pf_inits_tree = jax.tree_util.tree_map(lambda x: x[pick_global], samples_flat)
    mark("stratified_scoring", stage_s=round(time.perf_counter() - t, 3),
         scored=int(subset_idx.shape[0]))

    t = time.perf_counter()
    pf_samples = {n: pf_inits_tree[n] for n in sampler.rvs} if isinstance(
        pf_inits_tree, dict) else {n: a for n, a in zip(sampler.rvs, pf_inits_tree)}
    samples_total = jax.tree_util.tree_map(lambda x: x.reshape((total,) + x.shape[2:]),
                                           jax.tree_util.tree_map(
                                               lambda x: x.reshape((C, pf_draws) + x.shape[1:]),
                                               samples_flat))
    from jax.flatten_util import ravel_pytree

    def _flatten_i(i):
        sliced = jax.tree_util.tree_map(lambda x: x[i], samples_total)
        flat, _ = ravel_pytree(sliced)
        return flat
    flat_mat = jax.vmap(_flatten_i)(jnp.arange(total))
    X = flat_mat - flat_mat.mean(axis=0, keepdims=True)
    covariance = (X.T @ X) / (total - 1) + 1e-6 * jnp.eye(X.shape[1])
    inverse_mass_matrix = jnp.linalg.inv(covariance)
    mark("covariance", stage_s=round(time.perf_counter() - t, 3))

    t = time.perf_counter()
    eps0 = sampler._run_step_size_tune(pf_samples, inverse_mass_matrix, seed=seed)
    mark("eps0", stage_s=round(time.perf_counter() - t, 3), eps0=float(eps0))

    t = time.perf_counter()
    (chees_states, chees_parameters), chees_info = sampler._run_chees(
        pf_samples, inverse_mass_matrix, eps0,
        chains=C, draws=W, seed=seed, progressbar=False)
    mark("warmup_scan", stage_s=round(time.perf_counter() - t, 3))

    t = time.perf_counter()
    dhmc_state, states_hist, infos_hist = sampler._run_dynamic_hmc(
        chees_states, chees_parameters, inverse_mass_matrix,
        chains=C, draws=D, seed=seed, progressbar=False)
    mark("sampling_scan", stage_s=round(time.perf_counter() - t, 3))

    print(f"[done] {model_name} C={C} D={D} W={W} pf_draws={pf_draws} stages={stages_done}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chains", type=int, required=True)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--pf-draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "runs", "WP5__memprobe.jsonl"))
    a = ap.parse_args()
    run_cfg(a.model, a.chains, a.draws, a.warmup, a.pf_draws, a.seed, a.out)


if __name__ == "__main__":
    main()

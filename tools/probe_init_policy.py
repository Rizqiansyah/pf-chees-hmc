"""P1: does chain-start dispersion matter for ChEES's ensemble tuning?

Why a probe and not a reading task: the literature lane found no published comparison of stratified
versus iid chain starts with MCMC outcomes; ChEES's own paper is silent on starts; and Pathfinder's
PS-IR step resamples WITH replacement, which its paper shows collapsing to 4 distinct draws out of
100 on mcycle_gp. So the crux is only answerable by measurement.

Four start policies, identical model, metric, seeds and pipeline downstream of the start:

  same_point     all chains at the highest-ELBO Pathfinder point -- zero-dispersion control
  iid            one random Pathfinder draw per chain (what the sampler does today)
  stratified     equally spaced picks along Pathfinder draws sorted by model log-density
  overdispersed  stratified, then each start pushed 2x away from the ensemble mean

Read: if same_point ties the others at every warmup length, dispersion does not feed the ensemble
tuner and item (4) is dead. If short warmups separate them, the ranking tells us which selection
rule to ship.

Everything after the start choice is production code: _run_step_size_tune, _run_chees,
_run_dynamic_hmc, with progressbars off.
"""
import os
import sys
import time

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, "tools")
import bench_ladder as bl                       # noqa: E402
import blackjax as bj                           # noqa: E402
from pfcheeshmc.sampler import PfCheesHMC       # noqa: E402

MODEL = os.environ.get("MODEL", "radon")
CHAINS = int(os.environ.get("CHAINS", "32"))
DRAWS = int(os.environ.get("DRAWS", "500"))
WARMUPS = [int(w) for w in os.environ.get("WARMUPS", "250,500,1000,2000").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43").split(",")]
PF_DRAWS = int(os.environ.get("PF_DRAWS", "512"))   # per chain; enough to stratify over


def as_dict(pytree):
    """Pathfinder state positions are tuples in value-var order; the logp wants a dict."""
    if isinstance(pytree, dict):
        return pytree
    return {n: v for n, v in zip(list(PF_RVS), pytree)}


def diagnostics(pos, chains, draws):
    """max R-hat and min bulk ESS over sampled variables; pos leaves are (chain, draw, ...)."""
    rh, es = [], []
    for leaf in jax.tree_util.tree_leaves(pos):
        a = np.asarray(leaf, dtype=float)
        if a.ndim == 2:
            a = a[:, :, None]
        a = a.reshape(a.shape[0], a.shape[1], -1)
        for j in range(a.shape[2]):
            v = a[:, :, j]
            if np.all(np.isfinite(v)):
                rh.append(float(np.max(az.rhat(v))))
                es.append(float(np.min(az.ess(v, method="bulk"))))
    return (max(rh) if rh else float("nan")), (min(es) if es else float("nan"))


def run(model, policy, warmup, seed):
    sampler = PfCheesHMC(model, seed=seed)
    global PF_RVS
    PF_RVS = sampler.rvs
    key = jax.random.key(seed)
    k_fix, k_pf = jax.random.split(key, 2)

    init_stack = sampler._pymc_adapt_diag_start(CHAINS, init="adapt_diag", tune=500, random_seed=seed)
    init_stack = sampler._ensure_finite_inits(init_stack, k_fix)
    x0_pf = tuple(init_stack[n][0] for n in sampler.rvs)

    # One Pathfinder fit shared by every policy, so the only thing that varies is the pick.
    pf_state, _ = bj.vi.pathfinder.approximate(rng_key=k_pf, initial_position=x0_pf,
                                               logdensity_fn=sampler.ld_fn)
    # sample() returns (samples, state); taking the whole tuple adds a state leaf to the
    # pool and silently widens it to 176 columns, which then fails the ravel check.
    pf_sample, _ = bj.vi.pathfinder.sample(k_pf, pf_state, CHAINS * PF_DRAWS)
    leaves = jax.tree_util.tree_leaves(pf_sample)
    pool = np.stack([np.concatenate([np.asarray(l[i]).reshape(-1) for l in leaves])
                     for i in range(len(leaves[0]))])
    ref = tuple(init_stack[n][0] for n in sampler.rvs)
    _, unravel_pool = ravel_pytree(ref)
    # Score the pool in ONE vmapped compiled call. Doing it row by row is 64,000 eager calls into
    # the jaxed logp, which measured over 900 s for a single policy -- the same host-dispatch trap
    # the sampler itself was fixed for in W1 and in the start check.
    score = jax.jit(jax.vmap(lambda f: sampler.ld_fn(as_dict(unravel_pool(f)))))
    logp = np.asarray(score(jnp.asarray(pool)))
    if os.environ.get("DEBUG_SHAPES") == "1":
        print(f"DEBUG pool={pool.shape} scored in one call; ref_sizes="
              f"{[int(np.asarray(v).size) for v in ref]}", flush=True)
    imm = jnp.asarray(np.linalg.inv(np.cov(pool.T) + 1e-6 * np.eye(pool.shape[1])))

    n, d = pool.shape
    rng = np.random.default_rng(seed)
    if policy == "same_point":
        sel = np.full(CHAINS, int(np.argmax(logp)))
    elif policy == "iid":
        sel = rng.integers(0, n, size=CHAINS)
    elif policy in ("stratified", "overdispersed"):
        order = np.argsort(logp)
        edges = np.linspace(0, n, CHAINS + 1).astype(int)
        sel = np.array([order[(edges[i] + edges[i + 1]) // 2] for i in range(CHAINS)])
    else:
        raise ValueError(policy)
    starts = pool[sel].astype(pool.dtype)
    if policy == "overdispersed":
        m = starts.mean(axis=0, keepdims=True)
        starts = m + 2.0 * (starts - m)

    # Rebuild (chains, ...) per-variable arrays through ravel_pytree's inverse rather than by hand:
    # radon has five 0-d scalar value vars among 175 dimensions, and manual shape arithmetic gets
    # those wrong (it did: "Sum of sizes 175 must be equal to dimension 0 of [176]").
    _, unravel = ravel_pytree(x0_pf)
    starts_tree = jax.vmap(unravel)(jnp.asarray(starts))
    pf_samples = {n: leaf for n, leaf in zip(sampler.rvs, jax.tree_util.tree_leaves(starts_tree))}

    t0 = time.perf_counter()
    eps0 = sampler._run_step_size_tune(pf_samples, imm, eps_probes=min(CHAINS, 32))
    states, params, _ = sampler._run_chees(pf_samples, imm, eps0, draws=warmup, seed=seed,
                                           progressbar=False)
    if os.environ.get("DEBUG_SHAPES") == "1":
        print(f"DEBUG warmup done in {time.perf_counter()-t0:.2f}s", flush=True)
    _, hist_states, hist_infos = sampler._run_dynamic_hmc(states, params, imm, draws=DRAWS,
                                                          chains=CHAINS, seed=seed, progressbar=False)
    wall = time.perf_counter() - t0

    rhat, ess_min = diagnostics(hist_states.position, CHAINS, DRAWS)
    steps = float(np.mean(np.asarray(hist_infos.num_integration_steps)))
    spread = float(np.mean(starts.std(axis=0)))
    print(f"RESULT model={MODEL} policy={policy:<14} warmup={warmup:<5} seed={seed} "
          f"max_rhat={rhat:8.4f} min_bulk_ess={ess_min:8.0f} mean_steps={steps:5.1f} "
          f"spread={spread:9.4f} wall={wall:6.2f}s", flush=True)


if __name__ == "__main__":
    PF_RVS = []
    model = (bl.build_radon(True) if MODEL == "radon"
             else bl.load_notebook_model(bl.REPO / "examples" / "8 Schools example.ipynb"))
    for policy in ("same_point", "iid", "stratified", "overdispersed"):
        for w in WARMUPS:
            for seed in SEEDS:
                try:
                    run(model, policy, w, seed)
                except Exception as exc:
                    if os.environ.get("RERAISE") == "1":
                        raise
                    print(f"FAIL policy={policy} warmup={w} seed={seed}: "
                          f"{type(exc).__name__}: {str(exc)[:140]}", flush=True)

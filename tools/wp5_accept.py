"""WP5 acceptance: the R1 regime on the 6 GB card.

Plan acceptance (verbatim): radon noncentered runs at C=1,024 x D=10 on the 6 GB card,
and the init stages cost under 2 s warm. This harness measures both, plus the largest
C that fits per model.

Two processes per configuration, never mixed:
  rep 0 (cold)  -- fresh process: the stage peaks row (peak_bytes_in_use is
                   process-cumulative, so the memory reading is only valid cold).
  rep 1 (warm)  -- the process warms the compiler with one full sample(), then
                   times the stage methods of a second sample(). The init-stage
                   wall (everything up to and including the step-size tune) is the
                   number the acceptance names.

Run under tools/gpu_run.sh; timed cells call require_gpu_lock().
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp

from _common import new_row, write_row, banner, require_gpu_lock, gpu_other_procs, peak_mem_bytes

STAGE_METHODS = ["_pymc_adapt_diag_start", "_ensure_finite_inits", "_run_pathfinder",
                 "_run_step_size_tune", "_run_chees", "_run_dynamic_hmc"]
# Everything up to and including the step-size tune is "init" per the plan.
INIT_METHODS = ["_pymc_adapt_diag_start", "_ensure_finite_inits", "_run_pathfinder",
                "_run_step_size_tune"]


def timed_sampler(sampler):
    times = {}
    for name in STAGE_METHODS:
        orig = getattr(sampler, name)
        def wrap(fn=orig, nm=name):
            def inner(*a, **k):
                t0 = time.perf_counter()
                out = fn(*a, **k)
                times[nm] = time.perf_counter() - t0
                return out
            return inner
        setattr(sampler, name, wrap(nm))
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chains", type=int, required=True)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--pf-draws", type=int, default=4000)
    ap.add_argument("--pf-pool-cap", type=int, default=0,
                    help="0 = uncapped (pre-WP5 behaviour); else min(chains*draws, cap)")
    ap.add_argument("--pm-chains", type=int, default=1)
    ap.add_argument("--score-chunk", type=int, default=1024)
    ap.add_argument("--rep", type=int, default=0, help="0 = cold memory row, 1 = warm timing row")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "runs", "WP5__accept.jsonl"))
    a = ap.parse_args()

    import tools.bench_ladder as bl
    from pfcheeshmc import PfCheesHMC

    model = bl.model_specs()[a.model]()
    pf_kwargs = dict(draws=a.pf_draws, pm_chains=a.pm_chains, score_chunk=a.score_chunk)
    if a.pf_pool_cap:
        pf_kwargs["pf_pool_cap"] = a.pf_pool_cap

    banner(model=a.model, cold=(a.rep == 0), wp="WP5", arm="accept")
    require_gpu_lock("WP5 acceptance cell")

    sampler = PfCheesHMC(model, seed=a.seed)
    d_flat = sum(v.size for v in jax.tree_util.tree_leaves(
        {n: jnp.zeros_like(model.initial_point()[n]) for n in model.value_vars}))

    row = new_row(wp="WP5", arm="accept", model=a.model, d=int(d_flat), chains=a.chains,
                  draws_per_chain=a.draws, warmup=a.warmup, seed=a.seed, rep=a.rep,
                  cold=(a.rep == 0),
                  kwargs=json.dumps({"pf_draws": a.pf_draws, "pf_pool_cap": a.pf_pool_cap or None,
                                     "pm_chains": a.pm_chains, "score_chunk": a.score_chunk},
                                    sort_keys=True))

    if a.rep == 0:
        import blackjax as bj
        key = jax.random.key(a.seed)
        key_fix, key_pf, key_samp, key_pick, key_choose = jax.random.split(key, 5)
        t0 = time.perf_counter()
        pf_samples, imm = sampler._run_pathfinder(chains=a.chains, seed=a.seed, **pf_kwargs)
        t_pf = time.perf_counter() - t0
        eps0 = sampler._run_step_size_tune(pf_samples, imm, seed=a.seed)
        t_eps = time.perf_counter() - t0 - t_pf
        (chees_states, chees_parameters), chees_info = sampler._run_chees(
            pf_samples, imm, eps0, chains=a.chains, draws=a.warmup, seed=a.seed,
            progressbar=False)
        t_warm = time.perf_counter() - t0 - t_pf - t_eps
        sampler._run_dynamic_hmc(chees_states, chees_parameters, imm,
                                 chains=a.chains, draws=a.draws, seed=a.seed,
                                 progressbar=False)
        t_samp = time.perf_counter() - t0 - t_pf - t_eps - t_warm
        peak = peak_mem_bytes()
        row["stage_s"] = {"pathfinder_all": round(t_pf, 3), "eps0": round(t_eps, 3),
                          "warmup": round(t_warm, 3), "sampling": round(t_samp, 3),
                          "init_total": round(t_pf + t_eps, 3)}
        row["peak_mem_bytes"] = peak
        row["wall_s"] = round(time.perf_counter() - t0, 2)
        row["note"] = ("C=1,024 fit: yes" if peak else None)
        print(f"[accept cold] {a.model} C={a.chains}: init={t_pf + t_eps:.2f}s "
              f"warmup={t_warm:.2f}s sampling={t_samp:.2f}s peak={peak / 2**20:,.0f} MiB",
              flush=True)
    else:
        # warm: one throwaway full run compiles everything, then time the stages
        sampler.sample(draws=a.draws, random_seed=a.seed, progressbar=False,
                       pathfinder_kwargs=dict(chains=a.chains, **pf_kwargs),
                       chees_kwargs=dict(draws=a.warmup, progressbar=False))
        sampler2 = PfCheesHMC(model, seed=a.seed)
        times = timed_sampler(sampler2)
        t0 = time.perf_counter()
        idata = sampler2.sample(draws=a.draws, random_seed=a.seed, progressbar=False,
                                pathfinder_kwargs=dict(chains=a.chains, **pf_kwargs),
                                chees_kwargs=dict(draws=a.warmup, progressbar=False))
        total = time.perf_counter() - t0
        init_wall = sum(times.get(m, 0.0) for m in INIT_METHODS)
        # _run_pathfinder includes the wrapped pymc/finite calls; sum them once each.
        row["stage_s"] = {m: round(times.get(m, 0.0), 3) for m in STAGE_METHODS}
        row["stage_s"]["init_total"] = round(
            times.get("_run_pathfinder", 0.0) + times.get("_run_step_size_tune", 0.0), 3)
        row["wall_s"] = round(total, 2)
        peak = peak_mem_bytes()
        row["peak_mem_bytes"] = peak
        n_div = int(idata.sample_stats["diverging"].values.sum())
        row["divs_per_1k"] = round(n_div / (a.chains * a.draws) * 1000.0, 3)
        print(f"[accept warm] {a.model} C={a.chains}: init={row['stage_s']['init_total']:.2f}s "
              f"total={total:.2f}s peak(cum)={peak / 2**20:,.0f} MiB divs/1k={row['divs_per_1k']}",
              flush=True)

    row["gpu_other_procs"] = gpu_other_procs()
    write_row(a.out, row)
    print(f"[row] {a.out} :: rep={a.rep} {a.model} C={a.chains}", flush=True)


if __name__ == "__main__":
    main()

"""Child process of tests/test_api_prune.py: `sample_pf_chees_hmc` on a set of cases, in any tree.

    python api_prune_run.py TREE MODELS_PY GROUP OUT.npz

TREE provides `pfcheeshmc/` (this tree, or a `git archive` of the tag `archive/pre-cleanup`, which still has the
experiment options this work package removed). MODELS_PY is `benchmarks/models.py` of the tree under test: the models
are built by it in both runs, so only the package differs. GROUP picks the cases:

    default   the four ladder models, seeds 1 and 2, no options (the default path)
    kept      the options the default path keeps, on 8 Schools (every `imm_source`, `pm_chains` 4 and C (the same start
              point, which 8 Schools' `adapt_diag` start gives every chain, and jittered ones), `init_source`, `pm_init`, `pf_pool_cap`, `score_chunk`, `ridge`, the Fisher knobs, `warmup_stop`, the ChEES knobs, `eps0_probes`)
              and on GEV (the start repair, which the GEV start at xi = 0 makes work: its knobs)

Every case is 16 chains, 5 draws and 20 warmup draws. OUT.npz holds, per case, every posterior and sample_stats array
(``CASE/posterior/NAME``, ``CASE/stats/NAME``), the numeric attributes of the InferenceData (``CASE/attrs/NAME``) and the
resolved config as a JSON string (``CASE/config``), and ``meta`` (floatX, x64, the case labels).
"""
import importlib.util
import json
import os
import sys

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
tree, models_py, group, out = sys.argv[1:5]
sys.path.insert(0, tree)

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytensor  # noqa: E402
from pfcheeshmc import sample_pf_chees_hmc  # noqa: E402

spec = importlib.util.spec_from_file_location("ladder_models", models_py)
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)

CHAINS, DRAWS, WARMUP = 16, 5, 20
LADDER = ("8schools", "gev", "logit_1000x25", "radon_noncentered")

# (label, model, seed, sample_pf_chees_hmc keyword arguments beyond the common ones)
CASES = []
if group == "default":
    for name in LADDER:
        for seed in (1, 2):
            CASES.append((f"{name}/seed{seed}", name, seed, {}))
elif group == "kept":
    for source in ("mc", "lbfgs_lowrank", "fisher_lowrank", "pf_closed"):
        CASES.append((f"8schools/imm_source={source}", "8schools", 1,
                      {"pathfinder_kwargs": {"imm_source": source}}))
    CASES += [
        ("8schools/fisher_knobs", "8schools", 1, {"pathfinder_kwargs": {
            "imm_source": "fisher_lowrank", "fisher_draws": 64, "max_rank": 3, "fisher_gamma": 1e-4, "fisher_cutoff": 1.5}}),
        ("8schools/pm_chains=4", "8schools", 1, {"pathfinder_kwargs": {"pm_chains": 4}}),
        ("8schools/pm_chains=C+jitter", "8schools", 2, {"pathfinder_kwargs": {
            "pm_chains": CHAINS, "pm_init": "jitter+adapt_diag"}}),
        ("8schools/pm_chains=4+lbfgs_lowrank", "8schools", 1, {"pathfinder_kwargs": {
            "pm_chains": 4, "imm_source": "lbfgs_lowrank"}}),
        ("8schools/init_source=adapt_diag", "8schools", 1, {"pathfinder_kwargs": {"init_source": "adapt_diag"}}),
        ("8schools/init_source=adapt_diag+pm_chains=4", "8schools", 1, {"pathfinder_kwargs": {
            "init_source": "adapt_diag", "pm_chains": 4}}),
        ("8schools/pm_chains=4+jitter", "8schools", 1, {"pathfinder_kwargs": {
            "pm_init": "jitter+adapt_diag", "pm_chains": 4, "pm_tune": 100, "pm_seed": 9}}),
        ("8schools/pf_pool_cap", "8schools", 1, {"pathfinder_kwargs": {"pf_pool_cap": 64, "draws": 100}}),
        ("8schools/pf_pool_cap=0", "8schools", 1, {"pathfinder_kwargs": {"pf_pool_cap": 0, "draws": 20}}),
        ("8schools/score_chunk", "8schools", 1, {"pathfinder_kwargs": {"score_chunk": 7}}),
        ("8schools/ridge", "8schools", 1, {"pathfinder_kwargs": {"ridge": 1e-3}}),
        ("8schools/eps0_probes", "8schools", 1, {"step_size_tune_kwargs": {"eps0_probes": 8}}),
        ("8schools/chees_knobs", "8schools", 1, {"chees_kwargs": {
            "draws": WARMUP, "adam_lr": 0.1, "target_accept": 0.8, "max_leapfrog_steps": 20, "seed": 5}}),
        ("8schools/warmup_stop", "8schools", 1, {"chees_kwargs": {
            "draws": WARMUP, "warmup_stop": {"block": 10, "window": 10, "rhat": 1.01}}}),
        ("8schools/warmup_stop_fires", "8schools", 1, {"chees_kwargs": {
            "draws": WARMUP, "warmup_stop": {"block": 10, "window": 10, "rhat": 1e9, "eps_stability": 1e9}}}),
        ("8schools/dynamic_hmc_seed", "8schools", 1, {"dynamic_hmc_kwargs": {"seed": 11}}),
        ("gev/start_repair_knobs", "gev", 1, {"pathfinder_kwargs": {
            "pm_chains": 4, "pm_init": "jitter+adapt_diag", "pf_jitter_eps": 1e-6, "pf_max_backtracks": 8,
            "pf_max_restarts": 8, "pf_restart_radius": 0.1}}),
        ("gev/pm_chains=C", "gev", 2, {"pathfinder_kwargs": {"pm_chains": CHAINS}}),
    ]
else:
    raise SystemExit(f"unknown group {group!r}")

arrays = {}
meta = {"floatX": pytensor.config.floatX, "x64": bool(jax.config.jax_enable_x64), "cases": [c[0] for c in CASES]}
for label, name, seed, extra in CASES:
    extra = {k: dict(v) for k, v in extra.items()}
    extra.setdefault("chees_kwargs", {}).setdefault("draws", WARMUP)
    idata = sample_pf_chees_hmc(
        ladder.build_model(name), draws=DRAWS, chains=CHAINS, random_seed=seed, progressbar=False,
        keep_untransformed=True, **extra)
    for part, tag in ((idata.posterior, "posterior"), (idata.sample_stats, "stats")):
        for k, v in part.data_vars.items():
            arrays[f"{label}/{tag}/{k}"] = np.asarray(v.values)
    for k, v in idata.attrs.items():
        if k != "config":
            arrays[f"{label}/attrs/{k}"] = np.asarray(v)
    arrays[f"{label}/config"] = np.array(idata.attrs["config"])
np.savez(out, meta=np.array(json.dumps(meta)), **arrays)

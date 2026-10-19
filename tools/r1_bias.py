#!/usr/bin/env python
"""WP3 — R1 evaluation harness: score many-short-chain runs by BIAS, not by split R-hat.

Why this exists: in the owner's R1 regime (C ~ 1,024 chains, D ~ 10 draws) split R-hat is not a
convergence signal. Cutting ONE converged 32x2,000 run into 1024x39, 512x20, 512x78 and 128x313
reproduces every "degraded" R-hat the repo reports (1.050 / 1.090 / 1.024 / 1.006), so those numbers
measured the estimator, not the sampler (plan §2b). The gate here is bias against a NUTS reference:

    b2(f) = (mean_ens f - mean_ref f)^2 / var_ref f     for f in {x_i, x_i^2}, unconstrained space

plus nested R-hat (arviz_stats.rhat_nested — library, never reimplemented) when superchains are on.

It drives the PUBLIC `sample_pf_chees_hmc`. The only surgery is wrapping two methods in this process:
_to_inference_data to return unconstrained samples, and _run_chees to capture the frozen step size,
the frozen trajectory length and the exact warmup gradient counts. That is A8's stated contract for
the legacy arm; the sampler's own code path and numbers are untouched.

Usage (timed cells must be under the GPU lock):
  tools/gpu_run.sh pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH=$WT \
      python $WT/tools/r1_bias.py --models 8schools --chains 1024 --draws 10 \
      --warmups 100,200,400 --seeds 1,2,3 --repeat 2 --arm legacy \
      --pf-draws 1 --init-pick iid --superchains 8 --timed \
      --out results/runs/WP3__selftest.jsonl

G-R1 (plan §6): pass = b2_max < 0.01 on every seed with C*D >= 2,000. b2_max has its own Monte-Carlo
floor, max_f (mcse_f^2 / var_ref_f) * 2 ln(2d), which is printed with every cell: a cell whose floor
is above 0.005 is NOT RESOLVABLE, not a pass.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# A8 stage key -> the PfCheesHMC methods that make it up. _run_pathfinder NESTS the two init
# methods, so `init` is timed separately and subtracted from it (the double-count that made an
# earlier stage table add to more than the total).
STAGES = {
    "init": ["_pymc_adapt_diag_start", "_ensure_finite_inits"],
    "pathfinder": ["_run_pathfinder"],
    "eps0": ["_run_step_size_tune"],
    "warmup": ["_run_chees"],
    "sampling": ["_run_dynamic_hmc"],
    "postproc": ["_to_inference_data"],
}

INSTR = {"t": {}, "chees": {}}


# ---------------------------------------------------------------- references (WP0.2)

def _walk(obj, trail=()):
    """Yield (key_path, array) for every array in a container, 0-d included.

    0-d matters: a scalar variable's mean/var/mcse aggregates are stored as 0-d arrays, and skipping
    them makes a perfectly good reference file look like it is missing its statistics.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, trail + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, trail + (str(i),))
    elif hasattr(obj, "shape") and hasattr(obj, "ndim"):
        yield trail, obj


def build_model(model_name):
    """Build a ladder model and undo bench_ladder's global sampling stubs.

    `load_notebook_model` executes the example notebook with `pm.sample`,
    `pm.sample_prior_predictive`, `pm.sample_posterior_predictive` and
    `pfcheeshmc.sample_pf_chees_hmc` replaced by a stub (so loading a notebook does not sample) and
    **never restores them**. In the same process, everything that samples afterwards silently gets a
    one-draw dummy InferenceData. Observed twice: it made `az.rhat(idata.posterior)` raise on a
    2-variable stub, and merely importing this module's model list into a pytest run broke
    tests/test_rng_stage_keys.py by ordering alone. Snapshot, build, restore.
    """
    import bench_ladder as bl
    import pymc as pm
    import pfcheeshmc

    keep = (pm.sample, pm.sample_prior_predictive, pm.sample_posterior_predictive,
            pfcheeshmc.sample_pf_chees_hmc)
    try:
        return bl.model_specs()[model_name]()
    finally:
        (pm.sample, pm.sample_prior_predictive, pm.sample_posterior_predictive,
         pfcheeshmc.sample_pf_chees_hmc) = keep
        assert pfcheeshmc.sample_pf_chees_hmc is keep[3], "sampling stub was not restored"


def value_var_names(model_name):
    """Unconstrained variable names of a ladder model, in value_vars order (= Pathfinder's order).

    The scoring space and the WP1 ordering bug both live in this axis, so it is read from the model
    rather than from any stored file, and it is deliberately NOT sorted: the kernel's dict-flatten
    order is sorted, and the difference between the two is the bug WP1 fixes.
    """
    return [v.name for v in build_model(model_name).value_vars]


def _key_parts(key):
    """Split WP0.2's flat-npz key convention into (space, variable name).

    ~/pf-runs/references/<model>.npz stores `u__<name>` for unconstrained value vars, `x__<name>`
    for the constrained/problem-space variables, `mean__` / `var__` / `mcse_x__` / `mcse_x2__` for
    per-variable stats and `meta__*` for provenance. Names can themselves end in `__` (PyMC
    transformed names), so split on the FIRST `__` and only when the head is a known prefix.
    """
    head, sep, rest = str(key).partition("__")
    if sep and head in {"u", "x", "mean", "var", "mcse_x", "mcse_x2", "meta"}:
        return {"u": "unconstrained", "x": "constrained"}.get(head, "stat"), rest
    return "unknown", str(key)

def load_reference(model, names, require_stats=True):
    """{variable name: (n_draws, n_leaves)} for the model's value vars, from WP0.2's npz.

    Selection is by *variable name*, because the unconstrained names are exactly the model's value
    vars, and that holds for both flat-key (`u__name`) and nested-dict containers. An array under an
    explicit unconstrained label wins over an unlabelled one with the same name.

    `require_stats` enforces the WP0.2 contract on the stored aggregates: mean/var/mcse_x/mcse_x2 for
    a variable with vector shape must carry that shape, not a single number collapsed over the
    parameter axis. This harness recomputes those statistics from the sample arrays, so a collapsed
    file would NOT corrupt its own b2 -- but later WPs read the stored numbers, and a silent scalar
    for `theta` (8 of 10 parameters on 8 Schools) or `alpha_raw`/`beta_raw` (170 of 175 on radon NC)
    is exactly the kind of defect that has to fail at load time. Found live on 2026-10-19: the first
    reference set stored `mean__theta` as a scalar.
    """
    from _common import reference_path

    path = reference_path(model)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no reference at {path}; G-R1 cannot be evaluated without it (plan §6). Build it with "
            f"tools/make_reference.py {model} (WP0.2). Do not substitute a shorter run."
        )
    z = np.load(path, allow_pickle=True)
    root = z["data"].item() if ("data" in z.files and z["data"].ndim == 0) \
        else {k: z[k] for k in z.files}

    out, stats = {}, {}
    for trail, arr in _walk(root):
        space, name = _key_parts(trail[-1])
        if name in names and space not in ("stat", "constrained") and np.ndim(arr) >= 1:
            a = np.asarray(arr, dtype="float64")
            # WP0.2 stores (chains, draws, *shape): fold the first two axes into draws, the rest
            # into leaves. A 1-D array is already (n,) for one scalar leaf.
            a = a.reshape(-1, 1) if a.ndim == 1 else a.reshape(int(np.prod(a.shape[:2])), -1)
            unc = space == "unconstrained" or any(
                ("unconstr" in t) or ("value_var" in t) for t in trail[:-1])
            if name not in out or unc:
                out[name] = a
        elif space == "stat" and name in names:
            stats.setdefault(name, {})[str(trail[-1]).partition("__")[0]] = np.asarray(arr)
    if not out:
        raise KeyError(f"{path} has none of {sorted(names)}; top keys: "
                       f"{sorted({t for tr, _ in _walk(root) for t in tr[:1]})[:12]}")
    missing = sorted(set(names) - set(out))
    if missing:
        raise KeyError(f"{path} is missing unconstrained draws for {missing}; present: "
                       f"{sorted(out)}")
    if require_stats:
        bad = []
        for name, a in sorted(out.items()):
            n_leaf = a.shape[1]
            # A scalar variable is one leaf stored as a 0-d aggregate: () and (1,) are the same thing.
            want = () if n_leaf == 1 else a.shape[1:]
            got = stats.get(name, {})
            for kind in ("mean", "var", "mcse_x", "mcse_x2"):
                if kind not in got:
                    bad.append(f"{name}: no key {kind}__{name}")
                elif n_leaf > 1 and tuple(np.shape(got[kind])) != tuple(want):
                    bad.append(f"{kind}__{name} has shape {np.shape(got[kind])}, "
                               f"sample leaf shape {tuple(want)} "
                               f"({n_leaf} leaves collapsed to one)")
                elif n_leaf == 1 and np.asarray(got[kind]).size != 1:
                    bad.append(f"{kind}__{name} has {np.asarray(got[kind]).size} values for 1 leaf")
        if bad:
            raise ValueError(
                f"{path} violates the WP0.2 statistics contract (aggregates must keep the parameter "
                f"axis, not be collapsed over it):\n  " + "\n  ".join(bad[:12]) +
                "\n  Regenerate the file with tools/make_reference.py, or pass "
                "--allow-collapsed-reference-stats to score anyway (this harness recomputes the "
                "statistics from the sample arrays, so its own b2 is unaffected -- the flag records "
                "the provenance in the row).")
    return out


def flatten_run(by_name, chains, draws):
    """{name: (chains, draws, *shape)} -> ((chains*draws, leaves), labels), sorted by name.

    The shape is asserted, never inferred. A 2-D array is ambiguous -- it is either (chain, draw) for
    a SCALAR value variable or (rows, leaves) for a pre-flattened reference -- and guessing wrong
    silently under-samples exactly the scalar parameters: all three of GEV, and 5 of radon NC's 7
    (sigma_y_log__, sigma_alpha_log__, sigma_beta_log__, mu_alpha, mu_beta). Because b2_max is a max
    over leaves, an under-sampled scalar simply stops being the maximum: the number stays plausible
    and the gate stops covering those parameters.
    """
    cols, labels = [], []
    for name in sorted(by_name):
        a = np.asarray(by_name[name], dtype="float64")
        if a.ndim == 1:
            if a.size != chains * draws:
                raise ValueError(f"{name}: 1-D run array has {a.size} entries, "
                                 f"expected chains*draws = {chains * draws}")
            a = a.reshape(chains * draws, 1)
        else:
            if a.shape[0] != chains or a.shape[1] != draws:
                raise ValueError(f"{name}: expected (chains={chains}, draws={draws}, ...), "
                                 f"got {a.shape}")
            a = a.reshape(chains * draws, -1)
        cols.append(a)
        labels += [name] if a.shape[1] == 1 else [f"{name}[{j}]" for j in range(a.shape[1])]
    return np.concatenate(cols, axis=1), labels


def flatten_ref(by_name):
    """{name: (rows, leaves)} from WP0.2 -> ((rows, total_leaves), labels). All variables must
    carry the same number of reference rows (8 chains x 10,000 draws = 80,000)."""
    cols, labels = [], []
    rows = None
    for name in sorted(by_name):
        a = np.asarray(by_name[name], dtype="float64")
        if a.ndim != 2:
            raise ValueError(f"reference {name}: expected (rows, leaves), got {a.shape}")
        if rows is None:
            rows = a.shape[0]
        elif a.shape[0] != rows:
            raise ValueError(f"reference row counts disagree: {name} has {a.shape[0]}, others {rows}")
        cols.append(a)
        labels += [name] if a.shape[1] == 1 else [f"{name}[{j}]" for j in range(a.shape[1])]
    return np.concatenate(cols, axis=1), labels


# ---------------------------------------------------------------- scoring

def argmax_name(labels, k):
    """Name the leaf behind bias()'s index, which runs over [x_1..x_d, x^2_1..x^2_d] (2d entries).

    Indexing `labels` (d entries) with it is an IndexError when k >= d and, worse, names the wrong
    parameter whenever the x^2 half wins for an early leaf.
    """
    d = len(labels)
    return f"{labels[k % d]} ({'x' if k < d else 'x^2'})"


def bias(ens, ref, mcse_mean, mcse_mean_sq):
    """b2_max over {x, x^2}, its argmax leaf, and the MC noise floor. Pure numpy: testable on CPU."""
    d = ens.shape[1]
    if ref.shape[1] != d:
        raise ValueError(f"reference has {ref.shape[1]} leaves, run has {d}")
    den = np.maximum(ref.var(0, ddof=1), 1e-300)
    den2 = np.maximum((ref ** 2).var(0, ddof=1), 1e-300)
    b2m = (ens.mean(0) - ref.mean(0)) ** 2 / den
    b2v = ((ens ** 2).mean(0) - (ref ** 2).mean(0)) ** 2 / den2
    floor = max(float(np.max(np.asarray(mcse_mean, dtype="float64") ** 2 / den)),
                float(np.max(np.asarray(mcse_mean_sq, dtype="float64") ** 2 / den2))) \
        * 2.0 * float(np.log(2.0 * d))
    allb = np.concatenate([b2m, b2v])
    return float(np.nanmax(allb)), int(np.nanargmax(allb)), floor, b2m, b2v


def g_r1(b2_max, floor, c_times_d):
    """G-R1 as a verdict string. Not resolvable is NOT a pass (plan §6)."""
    if c_times_d < 2000:
        return "not eligible (C*D < 2000)"
    if floor > 0.005:
        return f"not resolvable (floor {floor:.4f} > 0.005)"
    return "pass" if b2_max < 0.01 else "fail"


def nominal_length(integration_steps_fn, W, max_sampling_steps=1000):
    """Frozen L_nominal = T/eps, bracketed rather than estimated.

    The warmup's integration_steps_fn(i) = ceil(h_i * L) with h_i the Halton jitter, so each realised
    integer count contributes one bound: L in [(L_i-1)/h_i, L_i/h_i]. The intersection is about
    1/h_max ~ 1 wide, so the midpoint is pinned to roughly half an integration step -- that is the
    precision of the legacy number and it is stated rather than hidden. Only used for arms that do
    not publish `frozen_L_nominal` in idata.attrs (A8).
    """
    import blackjax.mcmc.dynamic_hmc as dh

    if W < 1:
        return None
    bits = int(np.ceil(np.log2(W + max_sampling_steps)))
    # halton_sequence needs a JAX integer array (it reads i.dtype for the bit width), so the counts
    # are evaluated with int32 scalars rather than Python ints.
    def steps(i):
        return float(np.asarray(integration_steps_fn(jnp.asarray(i, dtype=jnp.int32))))

    h = np.asarray(jax.vmap(lambda i: dh.halton_sequence(i, bits))(jnp.arange(W)), dtype="float64")
    L = np.asarray([steps(i) for i in range(W)], dtype="float64")
    keep = (h > 0) & (L > 0)   # a zero-jitter or zero-step row carries no bound
    if not keep.any():
        return None
    h, L = h[keep], L[keep]
    lo, hi = float(np.max((L - 1.0) / h)), float(np.min(L / h))
    return 0.5 * (lo + hi) if hi >= lo else None    # empty intersection -> refuse to guess


# ---------------------------------------------------------------- instrumentation

def instrument():
    """Timers per stage + capture of the frozen adaptation values (A8's legacy-arm contract)."""
    import pfcheeshmc.sampler as sm

    def make(meth):
        orig = getattr(sm.PfCheesHMC, meth)

        def inner(self, *a, **k):
            t0 = time.perf_counter()
            out = orig(self, *a, **k)
            try:                                   # else GPU work lands on whoever reads last
                jax.block_until_ready(out[0] if isinstance(out, tuple) else out)
            except Exception:
                pass
            INSTR["t"][meth] = INSTR["t"].get(meth, 0.0) + (time.perf_counter() - t0)
            if meth == "_run_chees":
                # chees_warmup_with_progress returns (states, parameters), warm_info, and
                # _run_chees forwards that as (states, parameters, warm_info): the third element is
                # the stacked warmup HMCInfo, NOT a (positions, info) pair.
                _, params, winfo = out
                steps = np.asarray(winfo.num_integration_steps, dtype="int64")
                if steps.ndim == 1:
                    steps = steps[:, None]
                INSTR["chees"] = {
                    "frozen_eps": float(np.asarray(params["step_size"]).reshape(-1)[0]),
                    "frozen_L_nominal": nominal_length(params["integration_steps_fn"], steps.shape[0]),
                    "warmup_steps_used": int(params.get("warmup_steps_used", steps.shape[0])),
                    "grads_warmup_per_chain": steps.sum(0).tolist(),
                    "warmup_divs": int(np.asarray(winfo.is_divergent).sum()),
                }
            return out
        inner.__name__ = f"timed_{meth}"
        return inner

    for meths in STAGES.values():
        for meth in meths:
            setattr(sm.PfCheesHMC, meth, make(meth))


def force_unconstrained():
    """_to_inference_data(..., keep_untransformed=True), without touching the library's signature."""
    import pfcheeshmc.sampler as sm

    orig = sm.PfCheesHMC._to_inference_data

    def inner(self, states_hist, infos_hist=None, **kw):
        kw["keep_untransformed"] = True
        return orig(self, states_hist, infos_hist, **kw)

    sm.PfCheesHMC._to_inference_data = inner


def score_ensemble(unc, ref, chains, draws, superchains, nested_limit):
    """b2 / nested R-hat for one run. `unc` is {name: (chains, draws, leaves)}."""
    import arviz_stats as AS

    ens, labels = flatten_run(unc, chains, draws)
    reff, rlabels = flatten_ref(ref)
    if labels != rlabels:
        raise KeyError(f"run/reference leaf mismatch: {labels[:4]} vs {rlabels[:4]} "
                       f"(wrong space or missing value var?)")
    stacked = np.concatenate(
        [np.asarray(unc[n], dtype="float64").reshape(chains, draws, -1) for n in sorted(unc)],
        axis=2)                                                        # (C, D, d)
    d = stacked.shape[2]
    b2_max, k, floor, _, _ = bias(
        ens, reff,
        np.asarray(AS.mcse(stacked, method="mean", chain_axis=0, draw_axis=1)),
        np.asarray(AS.mcse(stacked ** 2, method="mean", chain_axis=0, draw_axis=1)))
    argmax = argmax_name(labels, k)
    nested = None
    if superchains and superchains > 1 and draws >= 2:
        ids = [i // (chains // superchains) for i in range(chains)]
        nested = float(np.asarray(AS.rhat_nested(stacked, superchain_ids=ids,
                                                 chain_axis=0, draw_axis=1)).max())
    return {"labels": labels, "d": int(d), "b2_max": b2_max,
            "b2_argmax": argmax, "b2_floor": floor,
            "nested_rhat_max": nested,
            "nested_pass": None if nested is None else bool(nested < nested_limit)}


def run_cell(bl, model_name, args, W, seed, rep, cold):
    from _common import banner, new_row, require_gpu_lock, stages, write_row

    if args.timed:
        require_gpu_lock(f"R1 timed cell {model_name} W={W}")
    import bench_ladder as bl
    import pfcheeshmc

    # A stubbed public sampler returns a 1-draw InferenceData instantly with no divergences, which
    # would look like an outstanding result instead of a failure. bench_ladder used to leak its
    # notebook-loading stubs for the whole process (fixed on pf-r1); this is the per-cell assertion.
    if pfcheeshmc.sample_pf_chees_hmc is not getattr(bl, "REAL_SAMPLE_PF", None):
        raise RuntimeError(
            "pfcheeshmc.sample_pf_chees_hmc is not bench_ladder.REAL_SAMPLE_PF "
            f"(got {getattr(pfcheeshmc.sample_pf_chees_hmc, '__name__', '?')}): a stub would "
            "produce a fake instant cell with no divergences")

    from pfcheeshmc import sample_pf_chees_hmc

    model = build_model(model_name)
    names = [v.name for v in model.value_vars]
    row = new_row(wp=args.wp, arm=args.arm, model=model_name, chains=args.chains,
                  draws_per_chain=args.draws, warmup=W, seed=seed, rep=rep, cold=cold,
                  superchains=args.superchains,
                  kwargs=json.dumps({"sampler_fn": "sample_pf_chees_hmc",
                                     "chains": args.chains, "draws": args.draws, "warmup": W,
                                     "pathfinder_kwargs": {"draws": args.pf_draws,
                                                           "init_pick": args.init_pick,
                                                           "superchains": args.superchains},
                                     "chees_kwargs": {"draws": W}, "progressbar": False,
                                     "ref_stats_check": "skipped"
                                     if args.allow_collapsed_reference_stats else "strict"},
                                    sort_keys=True))
    INSTR["t"], INSTR["chees"] = {}, {}
    t0 = time.perf_counter()
    try:
        idata = sample_pf_chees_hmc(
            model, chains=args.chains, draws=args.draws, random_seed=seed, progressbar=False,
            pathfinder_kwargs=dict(draws=args.pf_draws, init_pick=args.init_pick,
                                   superchains=args.superchains),
            chees_kwargs=dict(draws=W),
        )
        import arviz as az

        unc = {}
        for n in names:
            a = np.asarray(idata.posterior[n].values, dtype="float64")
            unc[n] = a.reshape(a.shape[0], a.shape[1], -1)
        sc = score_ensemble(unc, load_reference(model_name, names,
                                                require_stats=not args.allow_collapsed_reference_stats),
                            args.chains, args.draws, args.superchains, args.nested_limit)
        row.update(d=sc["d"], b2_max=sc["b2_max"], b2_floor=sc["b2_floor"],
                   b2_argmax=sc["b2_argmax"], nested_rhat_max=sc["nested_rhat_max"])

        # split R-hat through arviz exactly as tools/bench_ladder.py does, for comparability.
        # INFORMATION ONLY in R1: §2b shows it inflates on short chains that are not biased.
        rh = az.rhat(idata.posterior)
        finite = [float(np.max(v.values)) for v in rh.values() if np.all(np.isfinite(v.values))]
        row.update(max_rhat=round(max(finite), 4) if finite else None)

        steps = np.asarray(idata.sample_stats["num_integration_steps"].values, dtype="int64")
        steps = steps.reshape(steps.shape[0], steps.shape[1], -1)[..., 0]     # (C, D)
        divs = np.asarray(idata.sample_stats["diverging"].values, dtype="bool")
        attrs, ch = dict(idata.attrs), INSTR["chees"]
        row.update(
            frozen_eps=attrs.get("frozen_step_size", ch.get("frozen_eps")),
            frozen_L_nominal=attrs.get("frozen_L_nominal", ch.get("frozen_L_nominal")),
            steps_mean=float(steps.mean()) if steps.size else None,
            grads_per_chain_warmup=attrs.get(
                "warmup_grad_evals_per_chain",
                float(np.median(ch["grads_warmup_per_chain"])) if ch else None),
            grads_per_chain_sampling=float(np.median(steps.sum(1))) if steps.size else None,
            warmup_steps_used=attrs.get("warmup_steps_used", ch.get("warmup_steps_used")),
            warmup_divs=ch.get("warmup_divs"),
            divs_per_1k=float(divs.sum() / max(divs.size, 1) * 1000.0),
        )
        t = INSTR["t"]
        row.update(stage_s=stages(
            init=t.get("_pymc_adapt_diag_start", 0.0) + t.get("_ensure_finite_inits", 0.0),
            pathfinder=max(t.get("_run_pathfinder", 0.0)
                           - t.get("_pymc_adapt_diag_start", 0.0)
                           - t.get("_ensure_finite_inits", 0.0), 0.0),
            eps0=t.get("_run_step_size_tune", 0.0),
            warmup=t.get("_run_chees", 0.0),
            sampling=t.get("_run_dynamic_hmc", 0.0),
            postproc=t.get("_to_inference_data", 0.0)))
    except Exception:                    # a failed cell is a row, not a gap in the record
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        where = [l.strip() for l in tb if "line " in l and ("r1_bias" in l or "sampler.py" in l)]
        row.update(error=f"{tb[-1]} | {where[-1] if where else ''}"[:400],
                   wall_s=round(time.perf_counter() - t0, 2))
    else:
        from _common import peak_mem_bytes
        row.update(wall_s=round(time.perf_counter() - t0, 2),
                   peak_mem_bytes=peak_mem_bytes(),   # process-cumulative, read AFTER the run
                   cpu_contended=bool(args.cpu_contended),
                   # T1's objective is gradients per chain. A8 lists the two halves
                   # (grads_per_chain_warmup, grads_per_chain_sampling) but no sum, and a reporting
                   # task reading the file column-by-column has to add them. One column, same source.
                   grads_per_chain_total=(
                       (row["grads_per_chain_warmup"] or 0) + (row["grads_per_chain_sampling"] or 0)))

    from _common import gpu_other_procs
    row["gpu_other_procs"] = gpu_other_procs()   # A5: a foreign process voids a timing
    banner(model=model_name, cold=cold, wp=args.wp, arm=args.arm)
    write_row(args.out, row)
    verdict = g_r1(row["b2_max"] or np.inf, row["b2_floor"] or np.inf, args.chains * args.draws) \
        if row["error"] is None else "error"
    print(f"  [{model_name} {args.arm} W={W} seed={seed} rep={rep}] b2_max={row['b2_max']} "
          f"floor={row['b2_floor']} argmax={row['b2_argmax']} G-R1={verdict} "
          f"nRhat={row['nested_rhat_max']} eps={row['frozen_eps']} L={row['frozen_L_nominal']} "
          f"grads/chain={row['grads_per_chain_warmup']}+{row['grads_per_chain_sampling']} "
          f"divs/1k={row['divs_per_1k']} splitRhat={row['max_rhat']} wall={row['wall_s']}s "
          f"peak={row['peak_mem_bytes']} err={row['error']}", flush=True)
    return row


def run_starts(bl, model_name, args, rep, cold):
    """§2e's 'PF starts alone' row: the bias of the start ensemble before any sampling."""
    from _common import banner, new_row, write_row
    from pfcheeshmc.sampler import PfCheesHMC
    import arviz_stats as AS

    model = build_model(model_name)
    names = [v.name for v in model.value_vars]
    s = PfCheesHMC(model, seed=int(args.seeds.split(",")[0]))   # int: the CLI value is a string
    t_start = time.perf_counter()
    pos, imm = s._run_pathfinder(chains=args.chains, draws=args.pf_draws,
                                 init_pick=args.init_pick, superchains=args.superchains)
    # one start per chain => "draws" = 1 for the shape contract
    ens, labels = flatten_run({n: np.asarray(v) for n, v in pos.items()}, args.chains, 1)
    reff, rlabels = flatten_ref(load_reference(model_name, names))
    if labels != rlabels:
        raise KeyError(f"start/reference leaf mismatch {labels[:4]} vs {rlabels[:4]}")
    b2_max, k, _, _, _ = bias(ens, reff, np.zeros(ens.shape[1]), np.zeros(ens.shape[1]))
    row = new_row(wp=args.wp, arm="pf_starts", model=model_name, chains=args.chains,
                  draws_per_chain=1, warmup=0, seed=s.seed, rep=rep, cold=cold,
                  wall_s=round(time.perf_counter() - t_start, 2),
                  b2_max=b2_max, b2_argmax=argmax_name(labels, k), d=ens.shape[1],
                  kwargs=json.dumps({"chains": args.chains, "draws": args.pf_draws,
                                     "init_pick": args.init_pick,
                                     "superchains": args.superchains}, sort_keys=True))
    write_row(args.out, row)
    banner(model=model_name, cold=cold, wp=args.wp, arm="pf_starts")
    print(f"  [{model_name} pf_starts] b2_max={b2_max:.3f} at {row['b2_argmax']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="8schools,logit_1000x25")
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--warmups", default="100,200,400")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--arm", default="legacy")
    ap.add_argument("--wp", default="WP3")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "results", "runs",
                                                  "WP3__selftest.jsonl"))
    ap.add_argument("--pf-draws", type=int, default=1,
                    help="Pathfinder draws per chain; the pool is chains*this (4000 OOMs at C=1024)")
    ap.add_argument("--init-pick", default="stratified", choices=["stratified", "iid"])
    ap.add_argument("--superchains", type=int, default=0, help="K (superchains); 0 = off")
    ap.add_argument("--nested-limit", type=float, default=1.01)
    ap.add_argument("--cpu-contended", action="store_true",
                    help="tag rows taken while a CPU-heavy job (e.g. WP0.2 NUTS) was running")
    ap.add_argument("--timed", action="store_true", help="require the GPU lock (A5)")
    ap.add_argument("--allow-collapsed-reference-stats", action="store_true",
                    help="score against a reference whose stored aggregates are collapsed over the "
                         "parameter axis (WP0.2 defect). This harness recomputes them from the "
                         "sample arrays, so its b2 is unaffected; the row records the exemption.")
    ap.add_argument("--legacy", default="", help="checkout to import instead (e.g. ~/pf/legacy)")
    ap.add_argument("--starts-only", action="store_true", help="score the Pathfinder starts, no sampling")
    ap.add_argument("--check-references-only", action="store_true",
                    help="run the WP0.2 load-time contract check on --models and exit (no sampling)")
    args = ap.parse_args()
    args.superchains = args.superchains or None

    if args.check_references_only:
        for model_name in [m.strip() for m in args.models.split(",")]:
            names = value_var_names(model_name)
            try:
                ref = load_reference(model_name, names)
                leaves = sum(a.shape[1] for a in ref.values())
                print(f"[ref-guard] {model_name}: PASS  {len(names)} value vars, {leaves} leaves, "
                      f"{next(iter(ref.values())).shape[0]} reference draws")
            except Exception as exc:
                print(f"[ref-guard] {model_name}: FAIL  {type(exc).__name__}: {str(exc)[:300]}")
        return

    if args.legacy:
        sys.path.insert(0, args.legacy)
    import bench_ladder as bl
    instrument()
    force_unconstrained()
    print(f"[r1_bias] pfcheeshmc={sys.modules['pfcheeshmc'].__file__} "
          f"arm={args.arm} C={args.chains} D={args.draws} K={args.superchains}", flush=True)

    for rep in range(args.repeat):
        cold = rep == 0
        for model_name in [m.strip() for m in args.models.split(",")]:
            if args.starts_only:
                run_starts(bl, model_name, args, rep, cold)
                continue
            for W in [int(w) for w in args.warmups.split(",")]:
                for seed in [int(s) for s in args.seeds.split(",")]:
                    run_cell(bl, model_name, args, W, seed, rep, cold)


if __name__ == "__main__":
    main()

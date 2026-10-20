"""WP2 Deliverable 3: ChEES trajectory-length growth, metric x iterate.

Reproduction first (plan section 2c / review 2a targets, 32 chains, W=2,000):

    logistic 1000x25:  A precision,ma  eps ~0.0021  L ~136   (L@100/500/1000 ~ 6.6/37/92)
                       B cov,ma        eps ~0.66     L ~26
                       C cov,raw       eps ~0.66     L ~27    (L@100/500/1000 ~ 34/19/28)
                       D cov,raw,T0    t0 = pi/2
    8 Schools:         A eps ~0.18  L ~43 ;  C eps ~0.53  L ~19

Drives the PUBLIC warmup entry point (utils.chees_warmup_with_progress) on the
checkout tree; L@100/500/1000 come from the vendored adapter's per-step trace,
not from a monkeypatch. The metric variants isolate the two independent fixes:
A->B is orientation, A->C is orientation + raw iterate, C->D is the paper's
initial trajectory length on top.

Rows: results/runs/WP2__lgrowth.jsonl (pf-r1/v1 schema). REPEAT=2: rep 0 cold,
rep 1 warm; decisions use warm.

Env knobs: MODELS (default both), N, C, SEED, OUT.
"""
import os, sys, time

WT = os.environ.get("WT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "tools"))
os.chdir(WT)

import numpy as np
import jax
import jax.numpy as jnp
import optax

import _common as C
import bench_ladder as bl
from pfcheeshmc.sampler import PfCheesHMC
from pfcheeshmc.utils import chees_warmup_with_progress

jax.config.update("jax_enable_x64", True)

N = int(os.environ.get("N", "2000"))
CCH = int(os.environ.get("C", "32"))
SEED = int(os.environ.get("SEED", "43"))
OUT = os.environ.get("OUT") or os.path.join(WT, "results", "runs", "WP2__lgrowth.jsonl")
MODELS = os.environ.get("MODELS", "8schools,logit_1000x25").split(",")

VARIANTS = [
    ("A_prec_ma", "precision", "ma", None),
    ("B_cov_ma", "covariance", "ma", None),
    ("C_cov_raw", "covariance", "raw", None),
    ("D_cov_raw_t0", "covariance", "raw", np.pi / 2),
]


def main():
    C.require_gpu_lock("WP2 lgrowth")
    for name in MODELS:
        m = bl.model_specs()[name]()
        s = PfCheesHMC(m, seed=SEED)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            pos, imm_prec = s._run_pathfinder(chains=CCH, seed=SEED)
        imm_cov = jnp.linalg.inv(imm_prec)
        imm_cov = 0.5 * (imm_cov + imm_cov.T)
        d = sum(a.size // a.shape[0] for a in pos.values())
        print(f"[{name}] pathfinder done: d={d}", flush=True)

        for rep in (0, 1):
            for label, mkind, iterate, t0 in VARIANTS:
                imm = imm_prec if mkind == "precision" else imm_cov
                C.banner(model=name, wp="WP2", arm=f"wp2:{label}", cold=(rep == 0))
                row = C.new_row(
                    wp="WP2",
                    arm=f"wp2:{label}",
                    model=name,
                    d=d,
                    chains=CCH,
                    draws_per_chain=0,
                    warmup=N,
                    seed=SEED,
                    rep=rep,
                    cold=(rep == 0),
                    kwargs={
                        "metric": mkind,
                        "iterate": iterate,
                        "t0": None if t0 is None else float(t0),
                        "adam_lr": 0.05,
                    },
                    cpu_contended=False,
                )
                st = {"pathfinder": "shared", "eps0": None, "warmup": None}
                t0w = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        eps0 = float(s._run_step_size_tune(pos, imm, seed=SEED))
                    st["eps0"] = time.perf_counter() - t0w
                    key_warm = jax.random.fold_in(jax.random.key(SEED), 1)
                    t1 = time.perf_counter()
                    (states, params), warm = chees_warmup_with_progress(
                        s.ld_fn,
                        key_warm,
                        pos,
                        eps0,
                        optax.adam(0.05),
                        N,
                        imm,
                        progressbar=False,
                        iterate=iterate,
                        t0=t0,
                    )
                    jax.block_until_ready(params["frozen_step_size"])
                    st["warmup"] = time.perf_counter() - t1
                    tr = warm.trace
                    L = tr.trajectory_length / tr.step_size
                    checkpoints = {c: float(L[c - 1]) for c in (100, 500, 1000) if c <= N}
                    row.update(
                        wall_s=st["warmup"],
                        stage_s=C.stages(**st),
                        frozen_eps=float(params["frozen_step_size"]),
                        frozen_L_nominal=float(params["frozen_L_nominal"]),
                        steps_mean=float(np.mean(L)),
                        warmup_divs=int(np.sum(warm.info.is_divergent)),
                        kwargs=row["kwargs"] | {
                            "eps0": eps0,
                            "L_at": checkpoints,
                            "eps_at": {c: float(tr.step_size[c - 1]) for c in (100, 500, 1000) if c <= N},
                            "divs_at": {c: int(np.sum(warm.info.is_divergent[:c])) for c in (100, 500, 1000) if c <= N},
                        },
                    )
                    print(
                        f"[row] {name} {label} rep={rep} eps0={eps0:.4g} "
                        f"frozen_eps={float(params['frozen_step_size']):.4g} "
                        f"frozen_L={float(params['frozen_L_nominal']):8.2f} "
                        f"{' '.join(f'L@{c}={v:.2f}' for c, v in checkpoints.items())} "
                        f"divs={row['warmup_divs']} wall={st['warmup']:.1f}s",
                        flush=True,
                    )
                except Exception as e:  # record, keep going
                    row["error"] = f"{type(e).__name__}: {e}"
                    row["wall_s"] = time.perf_counter() - t0w
                    print(f"[row-err] {name} {label}: {row['error']}", flush=True)
                C.write_row(OUT, row)


if __name__ == "__main__":
    main()

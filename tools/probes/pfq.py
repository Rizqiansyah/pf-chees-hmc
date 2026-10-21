"""Pathfinder metric quality vs a CPU-NUTS reference posterior, in the unconstrained value space,
flattened in self.rvs order (the order the Pathfinder covariance is built in)."""
import os, sys, contextlib, io
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/home/akbar/Documents/Python packages/pf-chees-hmc/tools")
import numpy as np, jax, jax.numpy as jnp, pymc as pm
import bench_ladder as bl
from pfcheeshmc.sampler import PfCheesHMC
for name in sys.argv[1:]:
    m = bl.model_specs()[name](); s = PfCheesHMC(m, seed=42)
    with contextlib.redirect_stdout(io.StringIO()):
        pos, imm_prec = s._run_pathfinder(chains=32, seed=42)
    S_pf = np.linalg.inv(np.asarray(imm_prec)); S_pf = 0.5 * (S_pf + S_pf.T)
    with m:
        idata = bl.REAL_PM_SAMPLE(1000, tune=1000, chains=4, cores=4, random_seed=1, progressbar=False,
                          idata_kwargs={"include_transformed": True})
    post = idata.posterior
    cols = [post[n].values.reshape(4 * 1000, -1) for n in s.rvs]
    X = np.concatenate(cols, axis=1); S_post = np.cov(X, rowvar=False)
    r = np.diag(S_pf) / np.diag(S_post)
    w, V = np.linalg.eigh(S_pf); Wm = V @ np.diag(w ** -0.5) @ V.T
    lam = np.linalg.eigvalsh(Wm @ S_post @ Wm)
    dg = np.diag(S_pf) ** -0.5; lam_d = np.linalg.eigvalsh(dg[:, None] * S_post * dg[None, :])
    dpost = np.diag(S_post) ** -0.5; lam_post_diag = np.linalg.eigvalsh(dpost[:, None] * S_post * dpost[None, :])
    print(f"\n{name}: d={X.shape[1]}  cond(S_post)={np.linalg.cond(S_post):.0f}")
    print(f"  var ratio PF/post per dim: median {np.median(r):.3f}  p5 {np.percentile(r,5):.3f}  p95 {np.percentile(r,95):.3f}  min {r.min():.3g}  max {r.max():.3g}")
    print(f"  whitened by dense PF cov:   lam_min {lam.min():.3g} lam_max {lam.max():.3g}  cond {lam.max()/lam.min():.0f}")
    print(f"  whitened by diag of PF cov: cond {lam_d.max()/lam_d.min():.0f}")
    print(f"  whitened by TRUE posterior diag (ideal ensemble-diag metric): cond {lam_post_diag.max()/lam_post_diag.min():.0f}  lam_max {lam_post_diag.max():.2f}")
    if X.shape[1] > 10:
        worst = np.argsort(r)[:5]; names = [f"{n}[{i}]" for n in s.rvs for i in range(int(np.size(m.initial_point()[n])))]
        print("  most under-dispersed PF dims:", [(names[i], round(float(r[i]), 4)) for i in worst])

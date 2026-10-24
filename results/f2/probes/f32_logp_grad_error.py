"""F2k: error of the float32 log-density and gradient against float64, at reference-posterior points.

    python f32_logp_grad_error.py TREE MODEL OUT.npz            # run once with floatX=float64, once with floatX=float32
    python f32_logp_grad_error.py TREE MODEL --compare A64.npz B32.npz

Each run evaluates `lifted.lift_logp` (the log-density the sampler uses) and its gradient at 4,000 rows of the NUTS
reference (`~/pf-runs/references/MODEL.npz`, unconstrained space), taken in float64 on the host, and saves
position, lp and gradient. The comparison prints the relative error of lp and of the gradient by decile of the
distance of the point from the model's worst-conditioned coordinate (GEV: |xi_interval__|), and the worst rows.
"""
import os, sys
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tools")]
import numpy as np

if sys.argv[3] == "--compare":
    a, b = np.load(sys.argv[4]), np.load(sys.argv[5])
    lp_err = np.abs(b["lp"] - a["lp"]) / np.maximum(np.abs(a["lp"]), 1.0)
    g_err = np.linalg.norm(b["grad"] - a["grad"], axis=1) / np.maximum(np.linalg.norm(a["grad"], axis=1), 1e-12)
    print(f"rows {len(lp_err)}  lp rel err: median {np.median(lp_err):.2e} p99 {np.quantile(lp_err, .99):.2e} max {lp_err.max():.2e}")
    print(f"           grad rel err (2-norm): median {np.median(g_err):.2e} p99 {np.quantile(g_err, .99):.2e} max {g_err.max():.2e}")
    names = list(a["names"])
    for j, n in enumerate(names):
        col = a["x"][:, j] if a["x"].ndim == 2 else a["x"]
    # by |coordinate| of each scalar value variable: worst gradient error as a function of that coordinate
    x = a["x"]; err_j = np.abs(b["grad"] - a["grad"]) / np.maximum(np.abs(a["grad"]), 1.0)
    for j, lab in enumerate(a["labels"]):
        order = np.argsort(np.abs(x[:, j]))
        parts = np.array_split(order, 5)
        print(f"  {lab:18s} grad abs-or-rel err by quintile of |x|:",
              " ".join(f"{np.median(err_j[p, j]):.1e}/{err_j[p, j].max():.1e}" for p in parts))
    sys.exit()

import jax, pytensor, jax.numpy as jnp
import r1_bias
from pfcheeshmc import lifted, compiled
name, out = sys.argv[2:4]
model = r1_bias.build_model(name)
names = [v.name for v in model.value_vars]
ref = r1_bias.load_reference(name, names, require_stats=False)
rng = np.random.default_rng(0)
cols = [ref[n] for n in names]
idx = rng.choice(cols[0].shape[0], 4000, replace=False)
labels, parts = [], []
for n, c in zip(names, cols):
    c = c[idx]
    parts.append(c); labels += [n] if c.shape[1] == 1 else [f"{n}[{j}]" for j in range(c.shape[1])]
x = np.concatenate(parts, axis=1)
lg = lifted.lift_logp(model)
shapes = [model.initial_point()[n].shape for n in names]
dtype = jnp.zeros(()).dtype if not jax.config.jax_enable_x64 else jnp.float64
def split(row):
    out_, o = [], 0
    for s in shapes:
        k = int(np.prod(s)); out_.append(row[o:o + k].reshape(s)); o += k
    return tuple(out_)
def f(row):
    return lg.fn(split(row), lg.values)[0]
vg = jax.jit(jax.vmap(jax.value_and_grad(f)))
lp, grad = vg(jnp.asarray(x))
np.savez(out, x=x.astype(np.float64), lp=np.asarray(lp, dtype=np.float64), grad=np.asarray(grad, dtype=np.float64),
         names=np.array(names), labels=np.array(labels), dtype=str(lp.dtype))
print(name, "floatX", pytensor.config.floatX, "lp dtype", lp.dtype, "rows", x.shape)

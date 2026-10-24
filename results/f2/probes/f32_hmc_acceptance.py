"""F2k: HMC acceptance of the sampler's own kernel at the reference posterior, by step size, in float32 and float64.

    python f32_hmc_acceptance.py TREE MODEL

Starts 1,024 chains at rows of the NUTS reference (`~/pf-runs/references/MODEL.npz`, unconstrained space), takes one
dynamic-HMC step of L=10 leapfrog steps at each step size with the diagonal of the reference covariance as the metric,
and prints the mean acceptance probability, the harmonic mean over the non-divergent chains (what ChEES adapts the
step size on), the share of chains below 0.1 and the smallest acceptance. At the posterior a correct, well
conditioned density has acceptance near 1 for small step sizes in either dtype.
"""
import os, sys
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tools")]
import numpy as np, jax, jax.numpy as jnp, pytensor
import blackjax as bj
import r1_bias
from pfcheeshmc import lifted, compiled

name = sys.argv[2]
model = r1_bias.build_model(name)
names = [v.name for v in model.value_vars]
ref = r1_bias.load_reference(name, names, require_stats=False)
rng = np.random.default_rng(0)
C = 1024
idx = rng.choice(ref[names[0]].shape[0], C, replace=False)
x = {n: jnp.asarray(ref[n][idx].reshape((C,) + model.initial_point()[n].shape)) for n in names}
flat = np.concatenate([ref[n][idx].reshape(C, -1) for n in names], axis=1)
lg = lifted.lift_logp(model)
ld = compiled.LogDensity(lg.fn, tuple(names), lg.values)
sorted_names = sorted(names)                       # the kernel ravels a dict in sorted-key order
imm = jnp.asarray(np.concatenate([np.var(ref[n], axis=0).reshape(-1) for n in sorted_names]))
print("floatX", pytensor.config.floatX, "x64", jax.config.jax_enable_x64, "imm", np.asarray(imm).round(5))
for eps in (0.5, 0.2, 0.1, 0.05, 0.01, 0.002):
    kernel = bj.hmc(ld, eps, imm, 10)
    states = jax.vmap(kernel.init)(x)
    keys = jax.random.split(jax.random.key(1), C)
    _, info = jax.jit(jax.vmap(kernel.step))(keys, states)
    acc = np.asarray(info.acceptance_rate, dtype=np.float64)
    ok = ~np.asarray(info.is_divergent)
    with np.errstate(divide="ignore"):
        harmonic = 1.0 / np.mean(1.0 / acc[ok])           # the acceptance ChEES adapts the step size on
    low = np.flatnonzero(acc < 0.1)
    print(f"eps {eps:6.3f}  mean acceptance {np.nanmean(acc):.3f}  harmonic mean {harmonic:.4f}  "
          f"share < 0.1: {np.mean(acc < 0.1):.3f}  min acceptance {acc.min():.1e}  divergent {int(np.sum(info.is_divergent))}")
    for c in low[:4]:
        print(f"      chain {c}: acceptance {acc[c]:.1e} divergent {bool(info.is_divergent[c])}  start {dict(zip(names, flat[c].round(4)))}")

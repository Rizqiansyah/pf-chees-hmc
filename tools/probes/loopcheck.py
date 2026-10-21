import os, sys, time, contextlib, io
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
sys.path.insert(0, "/home/akbar/Documents/Python packages/pf-chees-hmc/tools")
from functools import partial
import numpy as np, jax, jax.numpy as jnp
import blackjax.mcmc.dynamic_hmc as dh
import bench_ladder as bl
from pfcheeshmc.sampler import PfCheesHMC
m = bl.model_specs()["8schools"](); s = PfCheesHMC(m, seed=42)
C = int(os.environ.get("C", "32")); Lnom = 40.0; eps = 0.2
ip = m.initial_point(); pos = {n: jnp.broadcast_to(jnp.asarray(ip[n]), (C,) + np.shape(ip[n])) + 0.0 for n in s.rvs}
d = sum(int(np.size(ip[n])) for n in s.rvs); imm = jnp.eye(d)
bits = 12; isf = lambda arg, L: jnp.asarray(jnp.ceil(dh.halton_sequence(arg, bits) * L), dtype=int)
step_fn = dh.build_kernel(next_random_arg_fn=lambda i: i + 1, integration_steps_fn=isf)
states = jax.vmap(lambda p: dh.init(p, s.ld_fn, 0))(pos)
def run(states, keys):
    def one(st, k):
        stp = partial(step_fn, logdensity_fn=s.ld_fn, step_size=eps, inverse_mass_matrix=imm, integration_steps_params=(Lnom,))
        ns, info = jax.vmap(stp)(jax.random.split(k, C), st)
        return ns, info.num_integration_steps.sum()
    return jax.lax.scan(one, states, keys)
keys = jax.random.split(jax.random.key(0), 500)
txt = jax.jit(run).lower(states, keys).as_text()
import re
print("while ops in StableHLO:", txt.count("stablehlo.while"), "| reduce(or) inside (batched predicate):", len(re.findall(r"stablehlo.or", txt)))
f = jax.jit(run); out = f(states, keys); jax.block_until_ready(out)
t = time.perf_counter(); out = f(states, keys); jax.block_until_ready(out); dt = time.perf_counter() - t
nlf = float(out[1].sum()) / C
print(f"C={C} XLA_FLAGS={os.environ.get('XLA_FLAGS','')!r}: {dt*1e3/500:.3f} ms/transition, {dt*1e6/nlf:.1f} us per leapfrog step (mean L {nlf/500:.1f})")

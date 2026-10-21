import os, sys
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/home/akbar/Documents/Python packages/pf-chees-hmc/tools")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
import bench_ladder as bl
from pfcheeshmc.sampler import PfCheesHMC
import blackjax as bj

for name in ["8schools", "gev", "radon_noncentered"]:
    m = bl.model_specs()[name]()
    s = PfCheesHMC(m, seed=0)
    ip = m.initial_point()
    sizes = {n: int(jnp.size(ip[n])) for n in s.rvs}
    print(name, "self.rvs:", [(n, sizes[n]) for n in s.rvs])
    print("   sorted :", sorted(s.rvs), "  SAME ORDER" if list(s.rvs) == sorted(s.rvs) else "  *** DIFFERENT ***")

# direct check: a diagonal metric built in tuple order, applied to a dict position
pos_dict = {"z_b": jnp.zeros(1), "a_vec": jnp.zeros(3)}   # creation order: z_b, a_vec
rvs = ["z_b", "a_vec"]
imm_tuple_order = jnp.array([100.0, 1.0, 1.0, 1.0])       # intent: z_b var 100, a_vec var 1
metric = bj.mcmc.metrics.default_metric(imm_tuple_order)
p = {"z_b": jnp.ones(1), "a_vec": jnp.zeros(3)}
print("tuple ravel:", ravel_pytree(tuple(p[n] for n in rvs))[0], " dict ravel:", ravel_pytree(p)[0])
print("KE with p only on z_b (intended 0.5*100=50):", metric.kinetic_energy(p))

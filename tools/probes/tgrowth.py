"""Probe: ChEES trajectory-length growth under (a) current code, (b) metric orientation fixed,
(c) orientation fixed + Adam on the raw iterate (paper Algorithm 1), MA used only for the freeze."""
import os, sys, time, types, json
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
sys.path.insert(0, "/home/akbar/Documents/Python packages/pf-chees-hmc/tools")
from functools import partial
import numpy as np, jax, jax.numpy as jnp, optax
import blackjax as bj, blackjax.mcmc.dynamic_hmc as dh
import bench_ladder as bl
from pfcheeshmc.sampler import PfCheesHMC

# variant of BlackJAX base(): one-line change -> kernel and Adam use the raw log-T iterate
src = open(bj.adaptation.chees_adaptation.__file__).read()
old = "new_trajectory_length = jnp.exp(new_log_trajectory_length_ma)"
assert src.count(old) == 1
raw_mod = types.ModuleType("chees_raw"); exec(compile(src.replace(old, "new_trajectory_length = jnp.exp(new_log_trajectory_length)"), "chees_raw", "exec"), raw_mod.__dict__)

def warmup(ld, pos, imm, eps0, N, base_fn, lr=0.05, C=None, seed=1, T0=None):
    bits = int(np.ceil(np.log2(N + 1000)))
    jit_gn = lambda i: dh.halton_sequence(i, bits)
    isf = lambda arg, L: jnp.asarray(jnp.ceil(jit_gn(arg) * L), dtype=int)
    step_fn = dh.build_kernel(next_random_arg_fn=lambda i: i + 1, integration_steps_fn=isf)
    init_a, upd_a = base_fn(jit_gn, lambda i: i + 1, optax.adam(lr), 0.651, 0.5, 1000)
    states = jax.vmap(lambda p: dh.init(p, ld, 0))(pos)
    a = init_a(0, eps0)
    if T0 is not None: a = a._replace(trajectory_length=jnp.asarray(T0, dtype=jnp.asarray(eps0).dtype))
    imm_diag = jnp.diag(imm) if imm.ndim == 2 else imm
    C = next(iter(pos.values())).shape[0]
    def one(carry, key):
        s, a = carry
        stp = partial(step_fn, logdensity_fn=ld, step_size=a.step_size, inverse_mass_matrix=imm,
                      integration_steps_params=(a.trajectory_length / a.step_size,))
        ns, info = jax.vmap(stp)(jax.random.split(key, C), s)
        a2 = upd_a(a, info.proposal.position, info.proposal.momentum, s.position, info.acceptance_rate, info.is_divergent, imm_diag)
        return (ns, a2), (a2.step_size, a2.trajectory_length, jnp.sum(info.is_divergent))
    (s, a), (eps, T, div) = jax.jit(lambda c, k: jax.lax.scan(one, c, k))((states, a), jax.random.split(jax.random.key(seed), N))
    L_final = float(jnp.exp(a.log_trajectory_length_moving_average - a.log_step_size_moving_average))
    eps_final = float(jnp.exp(a.log_step_size_moving_average))
    return np.asarray(eps), np.asarray(T), int(div.sum()), eps_final, L_final

N = int(os.environ.get("N", "2000")); C = int(os.environ.get("C", "32"))
for name in os.environ.get("MODELS", "8schools,logit_1000x25").split(","):
    m = bl.model_specs()[name](); s = PfCheesHMC(m, seed=42)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        pos, imm_prec = s._run_pathfinder(chains=C, seed=42)
    imm_cov = jnp.linalg.inv(imm_prec); imm_cov = 0.5 * (imm_cov + imm_cov.T)
    for label, imm, base_fn, T0 in [("A current (precision, MA-fed)", imm_prec, bj.adaptation.chees_adaptation.base, None),
                                    ("B cov, MA-fed", imm_cov, bj.adaptation.chees_adaptation.base, None),
                                    ("C cov, raw iterate", imm_cov, raw_mod.base, None),
                                    ("D cov, raw iterate, T0=pi/2", imm_cov, raw_mod.base, np.pi/2)]:
        eps0 = s._run_step_size_tune(pos, imm, seed=42)
        t = time.perf_counter()
        eps, T, div, epsF, LF = warmup(s.ld_fn, pos, imm, eps0, N, base_fn, T0=T0)
        L = T / eps
        cps = [100, 250, 500, 1000, 2000]
        trace = " ".join(f"L@{c}={L[c-1]:.1f}" for c in cps if c <= N)
        print(f"{name:14s} {label:30s} eps0={eps0:.3g} epsF={epsF:.3g} frozen L={LF:5.1f}  {trace}  divs={div}  ({time.perf_counter()-t:.1f}s)", flush=True)

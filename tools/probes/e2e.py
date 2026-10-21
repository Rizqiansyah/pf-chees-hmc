"""R1 regime probe: C chains x D draws, score by bias vs a CPU-NUTS reference (unconstrained space).
A = current pipeline semantics (precision metric, MA-fed ChEES). C = cov metric + raw-iterate ChEES."""
import os, sys, time, types, contextlib, io
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=-llapack -lblas -lcblas -lm")
sys.path.insert(0, "/home/akbar/Documents/Python packages/pf-chees-hmc/tools")
from functools import partial
import numpy as np, jax, jax.numpy as jnp, optax
from jax.flatten_util import ravel_pytree
import blackjax as bj, blackjax.mcmc.dynamic_hmc as dh
import bench_ladder as bl
from pfcheeshmc.sampler import PfCheesHMC
src = open(bj.adaptation.chees_adaptation.__file__).read()
old = "new_trajectory_length = jnp.exp(new_log_trajectory_length_ma)"
raw_mod = types.ModuleType("chees_raw"); exec(compile(src.replace(old, "new_trajectory_length = jnp.exp(new_log_trajectory_length)"), "chees_raw", "exec"), raw_mod.__dict__)
C = int(os.environ.get("C", "1024")); D = int(os.environ.get("D", "10"))

def run(ld, pos, imm, eps0, W, base_fn, seed):
    bits = int(np.ceil(np.log2(W + D + 2)))
    jgn = lambda i: dh.halton_sequence(i, bits)
    isf = lambda arg, L: jnp.asarray(jnp.ceil(jgn(arg) * L), dtype=int)
    step_fn = dh.build_kernel(next_random_arg_fn=lambda i: i + 1, integration_steps_fn=isf)
    init_a, upd_a = base_fn(jgn, lambda i: i + 1, optax.adam(0.05), 0.651, 0.5, 1000)
    imm_diag = jnp.diag(imm)
    def go(pos, key):
        states = jax.vmap(lambda p: dh.init(p, ld, 0))(pos); a = init_a(0, eps0)
        kw, ks = jax.random.split(key)
        def one(carry, k):
            s, a = carry
            stp = partial(step_fn, logdensity_fn=ld, step_size=a.step_size, inverse_mass_matrix=imm,
                          integration_steps_params=(a.trajectory_length / a.step_size,))
            ns, info = jax.vmap(stp)(jax.random.split(k, C), s)
            a2 = upd_a(a, info.proposal.position, info.proposal.momentum, s.position, info.acceptance_rate, info.is_divergent, imm_diag)
            return (ns, a2), (info.num_integration_steps[0], jnp.sum(info.is_divergent))
        (s, a), (Lw, dw) = jax.lax.scan(one, (states, a), jax.random.split(kw, W))
        eps = jnp.exp(a.log_step_size_moving_average); Ladj = jnp.exp(a.log_trajectory_length_moving_average - a.log_step_size_moving_average)
        def samp(s, k):
            stp = partial(step_fn, logdensity_fn=ld, step_size=eps, inverse_mass_matrix=imm, integration_steps_params=(Ladj,))
            ns, info = jax.vmap(stp)(jax.random.split(k, C), s)
            return ns, (ns.position, info.num_integration_steps[0], jnp.sum(info.is_divergent))
        _, (P, Ls, ds) = jax.lax.scan(samp, s, jax.random.split(ks, D))
        return P, Lw.sum() + Ls.sum(), dw.sum(), ds.sum(), eps, Ladj
    return jax.jit(go)(pos, jax.random.key(seed))

for name in os.environ.get("MODELS", "8schools,logit_1000x25").split(","):
    m = bl.model_specs()[name](); s = PfCheesHMC(m, seed=42)
    with m:
        ref = bl.REAL_PM_SAMPLE(5000, tune=1500, chains=8, cores=8, target_accept=0.95, random_seed=7,
                                progressbar=False, idata_kwargs={"include_transformed": True})
    R = np.concatenate([ref.posterior[n].values.reshape(40000, -1) for n in s.rvs], axis=1)
    m1, v1 = R.mean(0), R.var(0); m2, v2 = (R**2).mean(0), (R**2).var(0)
    with contextlib.redirect_stdout(io.StringIO()):
        pos, imm_prec = s._run_pathfinder(chains=C, seed=42, draws=int(os.environ.get("PFDRAWS", "4000")), init_pick=os.environ.get("PICK", "stratified"))
    imm_cov = jnp.linalg.inv(imm_prec); imm_cov = 0.5 * (imm_cov + imm_cov.T)
    pf_flat = np.asarray(jax.vmap(lambda *xs: ravel_pytree(tuple(xs))[0])(*[pos[n] for n in s.rvs]))
    b0 = max(np.max((pf_flat.mean(0) - m1)**2 / v1), np.max(((pf_flat**2).mean(0) - m2)**2 / v2))
    print(f"\n{name}: C={C} D={D}  Pathfinder starts alone: b2_max={b0:.3f}", flush=True)
    for label, imm, base_fn in [("A current", imm_prec, bj.adaptation.chees_adaptation.base), ("C cov+raw", imm_cov, raw_mod.base)]:
        eps0 = s._run_step_size_tune(pos, imm, seed=42)
        for W in [int(w) for w in os.environ.get("WS", "100,200,400,2000").split(",")]:
            res = []
            for seed in (1, 2, 3):
                t = time.perf_counter()
                P, grads, dw, ds, eps, Ladj = run(s.ld_fn, pos, imm, eps0, W, base_fn, seed)
                X = np.asarray(jax.vmap(jax.vmap(lambda *xs: ravel_pytree(tuple(xs))[0]))(*[P[n] for n in s.rvs])).reshape(C * D, -1)
                b2 = max(np.max((X.mean(0) - m1)**2 / v1), np.max(((X**2).mean(0) - m2)**2 / v2))
                res.append((b2, int(grads), int(ds), float(eps), float(Ladj), time.perf_counter() - t))
            r = np.array(res)
            print(f"  {label:10s} W={W:5d}: b2_max median {np.median(r[:,0]):.4f} (max {r[:,0].max():.4f})  grads/chain {int(np.median(r[:,1]))}  "
                  f"frozen eps {np.median(r[:,3]):.3g} L {np.median(r[:,4]):.1f}  sampling divs {int(r[:,2].sum())}  wall/run {np.median(r[:,5]):.1f}s(incl. compile)", flush=True)

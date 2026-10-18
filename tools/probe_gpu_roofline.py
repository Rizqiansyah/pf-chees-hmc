"""What actually limits a sampling step on this GPU: FLOPs, bytes, or latency.

Method: measure two ceilings on the device first (a big dense matmul for arithmetic, a big array
reduction for bytes-moved), then measure the real per-gradient cost of linear-Gaussian log-densities
at shapes that matter (8 Schools, radon d=175 n=919, and two larger-n radon variants), and a
zero-observation control at the same parameter dimension. Comparing achieved against ceiling is the
answer; comparing radon against the n=1 control at the same d separates "the parameter vector is
big" from "the data is big".

dtype is controlled by X64=1/0 because the package runs float64 and consumer GeForce/RTX cards pay a
large fp64 penalty; if fp32 is much cheaper at the same shape, part of the cost is the fp64 path.
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

X64 = os.environ.get("X64", "1") == "1"
jax.config.update("jax_enable_x64", X64)
DT = jnp.float64 if X64 else jnp.float32
ITEM = 8 if X64 else 4
NCHAINS = int(os.environ.get("NCHAINS", "32"))
REPEATS = 3

SHAPES = [(1, 175), (8, 10), (919, 175), (9190, 175), (91900, 175)]


def timeit(fn, *a):
    fn(*a)
    jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn(*a)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def emit(**kw):
    kw["dtype"] = "f64" if X64 else "f32"
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}, flush=True)


# ---- ceilings on this device, this dtype -------------------------------------------------------
big = jnp.ones((64 * 1024 * 1024,), dtype=DT)          # 512 MB (f64) / 256 MB (f32), read-only sum
t = timeit(lambda: jnp.sum(big))
bw_peak = big.size * ITEM / t / 1e9
del big
emit(arm="ceiling_bandwidth", gbytes_s=bw_peak)

m = jnp.ones((4096, 4096), dtype=DT)
t = timeit(lambda: m @ m)
flops_peak = 2 * 4096 ** 3 / t / 1e9
emit(arm="ceiling_flops", gflops=flops_peak)

# ---- per-gradient cost of a linear-Gaussian log-density ----------------------------------------
rng = np.random.default_rng(0)
for n, d in SHAPES:
    X = jnp.asarray(rng.normal(size=(n, d)), dtype=DT)
    y = jnp.asarray(rng.normal(size=(n,)), dtype=DT)
    b0 = jnp.zeros((NCHAINS, d), dtype=DT)

    def ld(b, X=X, y=y):
        return -0.5 * jnp.sum((y - X @ b) ** 2)

    vg = jax.jit(lambda P: jax.vmap(jax.value_and_grad(ld))(P))
    us = timeit(vg, b0) * 1e6
    bytes_read = NCHAINS * (n * d + n) * ITEM
    fl = NCHAINS * (2 * n * d + 3 * n)
    emit(arm="batched_grad_32chains", n=n, d=d, us=us,
         gb_per_grad=bytes_read / 1e9,
         gbytes_s=bytes_read / (us * 1e-6) / 1e9,
         pct_bw=100 * (bytes_read / (us * 1e-6)) / (bw_peak * 1e9),
         pct_flops=100 * (fl / (us * 1e-6)) / (flops_peak * 1e9))
    del X, y

# ---- chains scaling at the radon shape (latency vs parallelism) --------------------------------
n, d = 919, 175
X = jnp.asarray(rng.normal(size=(n, d)), dtype=DT)
y = jnp.asarray(rng.normal(size=(n,)), dtype=DT)


def ld(b):
    return -0.5 * jnp.sum((y - X @ b) ** 2)


for ch in (32, 64, 128):
    b0 = jnp.zeros((ch, d), dtype=DT)
    vg = jax.jit(lambda P: jax.vmap(jax.value_and_grad(ld))(P))
    us = timeit(vg, b0) * 1e6
    emit(arm="chains_scaling", chains=ch, n=n, d=d, us=us, us_per_chain=us / ch,
         gbytes_s=ch * (n * d + n) * ITEM / (us * 1e-6) / 1e9)

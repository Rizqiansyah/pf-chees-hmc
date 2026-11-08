# GPU notes

## Compiled Hessians can be wrong

:::{warning}
Set `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` before JAX starts.
:::

**What happened.** On one PyMC model (an upper-bounded GEV regression with a change point),
`jax.jit(jax.hessian(logp))` on the GPU returned a block of second derivatives that was wrong by up to a factor of
1,000,000. The same Hessian was right on the CPU, right on the GPU without `jit`, and right on the GPU with XLA's Triton GEMM
path switched off. A function that is linear in some inputs had a Hessian of 8.7e6 where the right answer is exactly 0.
JAX 0.12.0.1 (the latest release on 2026-10-28), CUDA 13, RTX A2000, driver 580.

**What is known and what is not.** Switching off Triton GEMM fixed it; five other XLA flags and
`JAX_DEFAULT_MATMUL_PRECISION=highest` did not. Why Triton gets it wrong (lost precision or a miscompiled kernel) is not
known, and no upstream report has been filed.

**Does it affect this sampler?** The sampler uses log-densities and gradients only; it computes no Hessian. On that model
the gradients were right: batched over 1,024 chains, as the sampler evaluates them, they matched the CPU to 3e-9 relative,
with and without Triton. That was checked for one model, not proven in general, so for a new model compare a few gradients
on the GPU against the CPU (`jax.jit(..., backend="cpu")`).

**What to do.** Set the flag in the shell or in the environment (pixi: `[activation.env]`):

```bash
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" python my_script.py
```

On that model it cost nothing measurable (a 1,024-chain gradient: 76.3 ms with Triton, 76.3 to 76.5 ms without). Anything
you build on top of the sampler that uses second derivatives (a Laplace approximation, a Newton optimiser, a Hessian-based
metric) must either run with this flag or compute the Hessian on the CPU.

## Runs at the same seed can differ on the GPU

On the GPU, two runs at the same seed can differ: different draws, step size and divergence count, but the same posterior
within Monte Carlo error.

- **The usual cause** is indexing in the model, `mu[group_idx]`. Its gradient is a scatter-add, which the GPU sums in no
  fixed order, so the gradient differs in the last bits between processes. Pathfinder's long L-BFGS path turns that into
  a different fit.
- The log density itself is reproducible. In jax 0.12 neither `--xla_gpu_deterministic_ops` nor
  `--xla_gpu_exclude_nondeterministic_ops` changes the gradient.
- **For bit-identical runs** (to replay one divergence, say), write the lookup as a product with a fixed one-hot matrix,
  `pt.dot(np.eye(n_groups)[group_idx], mu)`: the same model, and no scatter-add left.
- On the CPU, runs at the same seed are identical (`tests/test_golden.py`, which runs on the CPU only).

The metric the draws were made with is in `idata.attrs["frozen_inverse_mass_matrix"]` (flattened; shape in
`frozen_inverse_mass_matrix_shape`), so a divergence can be replayed with the sampler's own metric.

## Shared GPUs

JAX pre-allocates most of the GPU memory at import. On a shared card set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the pixi
environment does).

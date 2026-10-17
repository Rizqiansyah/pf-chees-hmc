# pf-chees-hmc
A Pathfinder-ChEES-HMC sampler using BlackJAX backend, compatible with PyMC model.

# ChEES-HMC vs. NUTS (or Why Use ChEES-HMC?)
The No-U-Turn sampler (NUTS) is a common and ubiquotous general purpose posterior sampling method that is the default if you use a PyMC model with continuous priors and likelihood.
It's a variant of the Hamiltonian Monte Carlo (HMC), with a main advantage that it avoids the need to tune the step size ($\epsilon$) and trajectory length ($L$) parameters in HMC, which is notoriously difficult and time consuming to do.
However, NUTS require substantial overhead in its computation, and it is not suitable for GPU acceleration.

ChEES-HMC, as proposed in [1] tune the HMC parameter automatically during the initial warm up phase.
HMC itself is suitable for GPU acceleration, and has less computational overhead compared to NUTS.
This allows posterior sampling efficiently using GPU(s).

NUTS is usually run over a small number of long chains, usually with the number of chains equal to the number of CPU cores available.
In contrast, ChEES-HMC can be run over a lot of short chains simultaneously in a GPU (or multiple GPUs).
This means that ChEES-HMC has the potential to sample much faster as the chains are shorter (for the same amount of effective sample size compared to NUTS).

# Implementation
The Pathfinder-ChEES-HMC sampler follows these steps:
1. Initialize the starting point using `adapt_diag` from `pymc` (the default initialization for NUTS in PyMC)
2. Use the Pathfinder algorithm (with some jitter) to find a better initial starting point and mass matrix.
3. Perform short HMC runs to tune the $\epsilon$ parameter.
4. Perform ChEES adaptation to tune the $\epsilon$, $L$, initial starting point, and other HMC parameters.
5. Run a Dynamic HMC using the tuned parameters from ChEES (Step 4) and mass matrix from the Pathfinder algorithm (Step 2)

# Installation

Requires Python 3.11+ (the floor set by BlackJAX 1.7). The repository ships a [pixi](https://pixi.sh) workspace that pins the GPU stack:

```
pixi install          # solve and create ./.pixi/envs/default
pixi run gpu-check    # must print: [CudaDevice(id=0)]
pixi run vram         # GPU name and memory
pixi run python       # or: pixi run jupyter lab
```

With pip instead, install the package and then a CUDA-enabled JAX:

```
pip install .
pip install -U "jax[cuda13]"      # GPU on Linux
```

# Versions and pins

- **jax / jaxlib >= 0.9, `cuda13` extra.** Needs driver >= 580 and an SM 7.5+ GPU. CUDA 12 wheels also work (driver >= 525, SM 5.2+) but are on the documented deprecation path. No system CUDA toolkit is required; the runtime ships inside the wheels.
- **blackjax >= 1.7.0.** This sampler calls BlackJAX internals (`chees_adaptation.base`, `dynamic_hmc.build_kernel`, `dynamic_hmc.halton_sequence`, `mcmc.dynamic_hmc.as_top_level_api`). BlackJAX 1.7 changed the ChEES `update()` signature (it now requires the diagonal inverse mass matrix) and replaced the kernel's `trajectory_length_adjusted` kwarg with `integration_steps_params`.
- **numpy is held below 2.5.** NumPy 2.5 removed `np.row_stack` and `np.trapz`; numba <= 0.66 registers overloads for them at import time, and numba >= 0.67 (which guards them) is outside PyTensor's `numba<=0.66` cap. Without this pin the resolver silently picks the pre-release `numba 0.63.0b1` and PyMC fails to import. See the comments in `pixi.toml`.

# Testing

```
pixi run python tests/test_inference_data.py
```

A few seconds, CPU only, no MCMC. It pins the `sample_stats` contract in `# Output` below.

# Use
See the `examples/` folder for some examples. In short
```
# Define your PyMC model here
with pymc.Model() as model:
  # Define prior ...
  # Define likelihood
  # Call the sampler
  trace = sample_pf_chees_hmc(draws=2000, chains=32, random_seed=42)

#Postprocess the trace with arviz, as usual
```

Pass `random_seed`: if it is omitted the seed defaults to the current date, so the run is not reproducible.

Stage-specific options are forwarded as dicts: `pathfinder_kwargs`, `step_size_tune_kwargs`, `chees_kwargs`, `dynamic_hmc_kwargs`.

For use with GPU sampling, see also the `examples/` folder.

# Output

`sample_pf_chees_hmc` returns an `arviz.InferenceData` with:

- `posterior` in **problem space**, including `pm.Deterministic` variables;
- `sample_stats` with exactly `lp`, `diverging`, `acceptance_rate`, `is_accepted`, `energy`, `num_integration_steps`. `lp` is the logdensity of the *sampled* state, matching what PyMC's own JAX samplers emit.

# Common Issues
1. If the sampler is not converging, try increasing the number of ChEES draws using the argument
```
sample_pf_chees_hmc(chees_kwargs={"draws": 10_000})
```
2. On a shared GPU, JAX pre-allocates most of the device memory at import. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the pixi env already does) so other users' jobs are not starved.
3. Divergence counts can differ by one between identical runs at the same seed. XLA does not guarantee bitwise-identical reductions, so a near-threshold divergence may flip; this is not a sampler change.


# References
[1] Matthew Hoffman, Alexey Radul, Pavel Sountsov "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo", in Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, PMLR 130:3907-3915, 2021. Available: https://proceedings.mlr.press/v130/hoffman21a.html

# pf-chees-hmc
A Pathfinder-ChEES-HMC sampler using BlackJAX backend, compatible with PyMC model.

# ChEES-HMC vs. NUTS (or Why Use ChEES-HMC?)
The No-U-Turn sampler (NUTS) is a common and ubiquotous general purpose posterior sampling method that is the default if you use a PyMC model with continuous priors and likelihood.
It's a variant of the Hamiltonian Monte Carlo (HMC), with a main advantage that it avoids the need to tune the step size ($\epsilon$) and trajectory length ($L$) parameters in HMC, which is notoriously difficult and time consuming to do.
However, NUTS require substantial overhead in its computation, and it is not suitable for GPU acceleration.

ChEES-HMC, proposed in [1], tunes the HMC step size ($\epsilon$) and trajectory length ($L$) automatically during the warm up phase. *ChEES* is the **Ch**ange in the **E**stimator of the **E**xpected **S**quare criterion that the method maximises (section 2.2 and Algorithm 1 of [1]). Note that ChEES adapts $\epsilon$ and $L$ only — it does not adapt the mass matrix, which is why this implementation takes that from Pathfinder.
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
  trace = sample_pf_chees_hmc(draws=2000, chains=128, random_seed=42)

#Postprocess the trace with arviz, as usual
```

Pass `random_seed`: if it is omitted the seed defaults to the current date, so the run is not reproducible.

**Chain count is the cheapest performance lever.** On an RTX A2000 the wall time per transition is flat from 32 to 256 chains (+2%), while ESS grows linearly with the number of chains, so more chains are nearly free until memory runs out. Measured on the 8 Schools example (2000 draws, 2,000 ChEES warmup, warm median of 5 seeds): 45,614 ESS/s at 32 chains and 134,898 ESS/s at 128. Sample histories are kept as `chains × draws × dim`, so raise `chains` with the model dimension in mind.

**How that compares to `pm.sample` on the same machine.** Best CPU configuration we could build on a 32-core box was 32 chains of NUTS: 37,494 ESS/s on 8 Schools and 6,733 on the GEV example. So the GPU is about 3.6x and 2.6x at 128 chains, and only ~1.2x at 32. For a small one-shot job CPU still finishes sooner (9.5 s versus 15.0 s), and its cold start is far lower — the GPU advantage is samples per second, not time-to-first-answer on tiny models.

**First run in a process is slower than later runs.** On the 8 Schools example at 32 chains: 48.3 s cold, 39.8 s cold with a persistent compile cache, 22 s warm in the same process. Setting `JAX_COMPILATION_CACHE_DIR` to a writable directory recovered 8.5 s of the 26 s compile cost at no cost to results (2.5 MB cache); the rest is JAX tracing, which that cache does not cover. Quote cold or warm, not both interchangeably.

**Turn the progress bars off in scripts:** `sample_pf_chees_hmc(..., progressbar=False)` removes the host callbacks from the compiled warmup and sampling scans. It is quieter and skips a per-step host round trip; it does **not** change compile cost — measured, a warm compile cache saves 6-7 s with the bars on or off, at ~640 XLA compiles either way.

**Warmup length: 2,000 is enough for these models; 10,000 was not.** Measured on both examples (2000 draws, 32 chains, warm median of 5 seeds, RTX A2000):

| model | warmup | wall | ESS/s | max R-hat | divs /1k draws | tail-ESS |
|---|---|---|---|---|---|---|
| 8 Schools | 1,000 | 10.4 s | 48,014 | 1.0011 | 0.008 | 30,227 |
| 8 Schools | 2,000 | 12.1 s | 39,624 | 1.0012 | 0.024 | 31,743 |
| 8 Schools | 10,000 | 22.4 s | 21,082 | 1.0014 | 0.008 | 32,436 |
| GEV | 1,000 | 12.1 s | 1,469 | **1.0680** | 0.039 | **583** |
| GEV | 2,000 | 16.2 s | 6,486 | 1.0052 | 0.125 | 8,163 |
| GEV | 10,000 | 50.9 s | 3,672 | 1.0009 | 0.438 | 24,048 |

Do not go as low as 1,000: on GEV the frozen trajectory length collapsed to 6.2 leapfrog steps, R-hat went to 1.068 and tail-ESS to 583 — ChEES had not learned the trajectory length. At 2,000 the posterior is indistinguishable from 10,000: worst per-variable difference 0.49 sigma (8 Schools) and 0.70 sigma (GEV), posterior sds within 0.6%. More chains also means less needed warmup, since ChEES pools statistics across all chains.

Stage-specific options are forwarded as dicts: `pathfinder_kwargs`, `step_size_tune_kwargs`, `chees_kwargs`, `dynamic_hmc_kwargs`.

For use with GPU sampling, see also the `examples/` folder.

# Output

`sample_pf_chees_hmc` returns an `arviz.InferenceData` with:

- `posterior` in **problem space**, including `pm.Deterministic` variables;
- `sample_stats` with exactly `lp`, `diverging`, `acceptance_rate`, `is_accepted`, `energy`, `num_integration_steps`. `lp` is the logdensity of the *sampled* state, matching what PyMC's own JAX samplers emit.

# Common Issues
1. If the sampler is not converging, try increasing the number of ChEES draws using the argument
```
sample_pf_chees_hmc(chees_kwargs={"draws": 2_000})
```
2. On a shared GPU, JAX pre-allocates most of the device memory at import. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the pixi env already does) so other users' jobs are not starved.
3. Divergence counts can differ by one between identical runs at the same seed. XLA does not guarantee bitwise-identical reductions, so a near-threshold divergence may flip; this is not a sampler change.


# References
[1] Matthew Hoffman, Alexey Radul, Pavel Sountsov "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo", in Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, PMLR 130:3907-3915, 2021. Available: https://proceedings.mlr.press/v130/hoffman21a.html

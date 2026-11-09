# pf-chees-hmc

**Pathfinder-initialised ChEES-HMC for PyMC models, on JAX and BlackJAX.** Thousands of short chains run in lock step,
which is what a GPU does well. Give it a PyMC model with continuous variables; get back an `arviz.InferenceData`.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![version](https://img.shields.io/badge/version-0.3.2-informational)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
[![docs](https://github.com/Rizqiansyah/pf-chees-hmc/actions/workflows/docs.yml/badge.svg)](https://rizqiansyah.github.io/pf-chees-hmc/)
![backend](https://img.shields.io/badge/backend-JAX%20%7C%20BlackJAX-orange)

```python
import pymc as pm
from pfcheeshmc import sample_pf_chees_hmc

with pm.Model() as model:
    ...                                   # priors and likelihood
    idata = sample_pf_chees_hmc(draws=10, chains=1024, random_seed=42)
```

> [!WARNING]
> **Compiled Hessians on the GPU can be wrong.** Set `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. On one model,
> `jax.jit(jax.hessian(logp))` on the GPU was wrong by up to a factor of 10^6 (JAX 0.12.0.1, CUDA 13); switching off
> Triton GEMM fixed it. The sampler uses gradients only, and they were right on that model, but anything you build on top
> that uses second derivatives is exposed. Details: [docs/gpu-notes.md](docs/gpu-notes.md).

## Highlights

- **PyMC in, ArviZ out.** `posterior` in problem space (deterministics included), `sample_stats` with `lp`, `diverging`,
  `energy`, `num_integration_steps` and more, and the resolved configuration in `idata.attrs`.
- **Case study: 19 to 76x faster than NUTS on a whole 56-core CPU** at the same effective sample size, on a stiff
  hierarchical extreme-value regression (five datasets; RTX A2000; same quality bar, quantiles within 0.3%)
  ([Case study](docs/case-study-gev-regression.md)).
- **Fast on a GPU.** One warm call against PyMC NUTS on 32 CPU cores, both at settings that pass the same bias test
  against a long NUTS reference: 15x faster on 8 Schools, 9.4x on a GEV model, 6.8x on a 1000 x 25 logistic regression,
  2.5 to 3.8x on non-centred radon (RTX A2000, [Performance](docs/performance.md)).
- **Compile once.** A new seed, new data or new prior values at the same shapes compile nothing, and compiled programs
  are kept on disk for the next process ([Compile once](docs/compile-and-cache.md)).
- **Tunable for stiff posteriors.** Pathfinder's L-BFGS settings, four metric sources, and opt-in metric re-estimation
  during warmup ([Metrics](docs/metrics.md)).
- **Reproducible.** On the CPU, runs at the same seed are identical. On the GPU they can differ in the last bits but agree
  within Monte Carlo error; writing an index lookup as a one-hot product made them identical on the model measured
  ([GPU notes](docs/gpu-notes.md)).

## Install

```bash
pip install .                 # CPU
pip install ".[cuda13]"       # NVIDIA GPU: driver >= 580 on Linux, SM 7.5 or newer
```

Python 3.11+, BlackJAX 1.7+. The repository also ships a [pixi](https://pixi.sh) workspace that pins the GPU stack
(`pixi install`, then `pixi run gpu-check`). More: [Installation](docs/installation.md).

## How it works

1. Starts from PyMC's `adapt_diag` initialisation, repaired where the log-density or its gradient is not finite.
2. Pathfinder: better starts and the mass matrix.
3. A short search for the initial step size.
4. ChEES warmup [1]: the step size and one trajectory length, shared by all chains.
5. HMC draws with those values frozen, then the transform back to problem space.

Because every chain does the same work at each step, 1,024 chains cost little more than a few. More:
[How it works](docs/how-it-works.md).

## Documentation

**Documentation: <https://rizqiansyah.github.io/pf-chees-hmc/>**, a Sphinx site built from [`docs/`](docs/index.md) on every
push to `main`. To build it locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

| page | what is in it |
|---|---|
| [Quick start](docs/quickstart.md) | arguments, seeds, what comes back, the example notebooks |
| [Choosing chains, draws and warmup](docs/choosing-settings.md) | the bias gate, warmup lengths that pass it, warmup options |
| [Metrics](docs/metrics.md) | metric sources, Pathfinder's L-BFGS settings, metric updates |
| [Performance](docs/performance.md) | against PyMC NUTS, with the commands |
| [Case study](docs/case-study-gev-regression.md) | a stiff extreme-value regression: one GPU against NUTS on 56 cores |
| [Compile once, and the disk cache](docs/compile-and-cache.md) | what compiles when; the on-disk cache |
| [float32](docs/float32.md) | opt-in; where it helps and where it silently fails |
| [GPU notes](docs/gpu-notes.md) | the Hessian warning, reproducibility, shared GPUs |
| [Troubleshooting](docs/troubleshooting.md) | not converging, divergences, slow first calls |
| [API](docs/api.md) | `sample_pf_chees_hmc`, the cache functions |
| [Design notes](docs/design.md) | for maintainers |

Example notebooks are in [`examples/`](examples/); the benchmark harness in [`benchmarks/`](benchmarks/README.md).

## Tests

```bash
JAX_PLATFORMS=cpu REFERENCE_FILES_OPTIONAL=1 pixi run python -m pytest tests -q
```

The suite runs on the CPU in one process: 555 passed, 2 skipped in 8 min 45 s (2026-11-09, shared host; the skips need
reference files built by `benchmarks/make_reference.py`).
`tests/test_golden.py` pins the draws of the default path on the benchmark models.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

## References

[1] M. Hoffman, A. Radul, P. Sountsov, "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo",
AISTATS, PMLR 130:3907-3915, 2021. <https://proceedings.mlr.press/v130/hoffman21a.html>

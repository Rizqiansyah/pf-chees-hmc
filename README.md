# pf-chees-hmc

**Pathfinder-initialised ChEES-HMC for PyMC models, on JAX and BlackJAX.** Thousands of short chains run in lock step,
which is what a GPU does well. Give it a PyMC model with continuous variables; get back an `arviz.InferenceData`.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![version](https://img.shields.io/badge/version-0.4.0-informational)
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

## What's new in 0.4

Every addition is opt-in and the defaults are those of 0.3.2. The draws differ from 0.3.2's only through the JAX update in
the lock (0.12; at rounding level in float64).

- **Diagnostics in the output.** `idata.attrs["diagnostics"]`: Pathfinder's L-BFGS iterations against its cap, and the
  divergences by the energy at the trajectory's end (left the support, arithmetic failure, step too large), with
  warnings for a capped L-BFGS, NaN divergences, and float32 matrix products in TF32 on Ampere and later GPUs
  ([Troubleshooting](docs/troubleshooting.md#divergences)).
- **Delayed rejection on divergence (opt-in).** `dynamic_hmc_kwargs={"delayed_rejection": a}` retries a divergent
  transition with the step divided by a, up to several stages, and stays exact (Modi, Barnett & Carpenter 2024); only the
  chains that diverged pay for it. A stage's proposal and its ghosts run in one loop, so the compile time grows about
  linearly with the stages (4 stages: 24.5 s on a GPU). Retry rules beyond divergences (experimental)
  ([Choosing settings](docs/choosing-settings.md#divergences-in-a-rare-region-delayed-rejection-opt-in)).
- **A Fisher-divergence metric at metric updates (experimental)**, from the chains' positions and gradients: large gains
  when the chains are few for the dimension ([Metrics](docs/metrics.md)).
- **Reading R-hat with many chains**: why R-hat above 1.01 is expected for slow coordinates with ~1,000 short chains, and
  the check that tells slow mixing from chains that disagree ([R-hat](docs/rhat-many-chains.md)).
- **Two warmup settings against run-to-run spread (experimental).** `chees_kwargs["acceptance_floor"]` (a floor on each
  chain's acceptance in the step size's statistic) and `chees_kwargs["trajectory_length"]` (a fixed trajectory length,
  which you choose for your model). Together, on an upper-bounded GEV regression: 1.7-6.9x the ESS per gradient and a
  run-to-run spread of 0.02-0.15 against 0.09-0.43 (sd of log), with more first-stage divergences, which delayed
  rejection resolved ([Choosing settings](docs/choosing-settings.md#run-to-run-spread-a-step-size-floor-and-a-fixed-trajectory-length-opt-in-experimental)).
- **An options reference**: every key of every stage, its default, and whether it is opt-in ([Options](docs/options.md)).
- **Several leapfrog steps per loop pass** (`leapfrog_unroll`): about 7% less sampling time on a GPU at 4 steps per
  pass, for a longer first compile ([Performance](docs/performance.md#several-leapfrog-steps-per-loop-pass-opt-in)).
- **Option values are checked before the first stage runs**: a bad value raises a `ValueError` at once, not after
  Pathfinder.
- `pfcheeshmc.__version__`; JAX 0.12 in the lock; `pixi run -e docs docs` builds the documentation.

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
- **Exact retries at a rare hard edge.** Opt-in delayed rejection retries a divergent transition with smaller steps, up to
  several stages, and stays exact (Modi, Barnett & Carpenter 2024); only the chains that diverged pay for it
  ([Choosing settings](docs/choosing-settings.md#divergences-in-a-rare-region-delayed-rejection-opt-in)).
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
| [Choosing chains, draws and warmup](docs/choosing-settings.md) | the bias gate, warmup lengths that pass it, warmup options, run-to-run spread, delayed rejection |
| [Metrics](docs/metrics.md) | metric sources, Pathfinder's L-BFGS settings, metric updates |
| [Reading R-hat with many chains](docs/rhat-many-chains.md) | why R-hat above 1.01 is expected for slow coordinates, and the check that tells why |
| [Performance](docs/performance.md) | against PyMC NUTS, with the commands; several leapfrog steps per loop pass |
| [Case study](docs/case-study-gev-regression.md) | a stiff extreme-value regression: one GPU against NUTS on 56 cores |
| [Compile once, and the disk cache](docs/compile-and-cache.md) | what compiles when; the on-disk cache |
| [float32](docs/float32.md) | opt-in; where it helps and where it silently fails |
| [GPU notes](docs/gpu-notes.md) | the Hessian warning, reproducibility, shared GPUs |
| [Troubleshooting](docs/troubleshooting.md) | not converging, divergences, slow first calls |
| [API](docs/api.md) | `sample_pf_chees_hmc`, the cache functions |
| [Options](docs/options.md) | every option of every stage, its default, and whether it is opt-in |
| [Design notes](docs/design.md) | for maintainers |

Example notebooks are in [`examples/`](examples/); the benchmark harness in [`benchmarks/`](benchmarks/README.md).

## Tests

```bash
JAX_PLATFORMS=cpu REFERENCE_FILES_OPTIONAL=1 pixi run python -m pytest tests -q
```

The suite runs on the CPU in one process: 690 passed, 2 skipped in 5 min 13 s (2026-11-12, 6 cores of a Ryzen 9 7900X;
the skips need reference files built by `benchmarks/make_reference.py`).
`tests/test_golden.py` pins the draws of the default path on the benchmark models.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

## References

[1] M. Hoffman, A. Radul, P. Sountsov, "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo",
AISTATS, PMLR 130:3907-3915, 2021. <https://proceedings.mlr.press/v130/hoffman21a.html>

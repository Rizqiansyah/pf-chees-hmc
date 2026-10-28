# pf-chees-hmc

A Pathfinder-ChEES-HMC sampler for PyMC models, built on JAX and BlackJAX. `sample_pf_chees_hmc` takes a PyMC model with continuous variables and returns an `arviz.InferenceData`.

> **WARNING: compiled Hessians on the GPU can be WRONG. Set `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`.**
>
> **What happened.** On one PyMC model (an upper-bounded GEV regression with a change point), `jax.jit(jax.hessian(logp))` on the GPU returned a block of second derivatives that was wrong by up to a factor of 1,000,000. The same Hessian was right on the CPU, right on the GPU without `jit`, and right on the GPU with XLA's Triton GEMM path switched off. A function that is linear in some inputs had a Hessian of 8.7e6 where the right answer is exactly 0. JAX 0.12.0.1 (the latest release on 2026-10-28), CUDA 13, RTX A2000, driver 580.
>
> **What is known and what is not.** Switching off Triton GEMM (`--xla_gpu_enable_triton_gemm=false`) fixed it; five other XLA flags and `JAX_DEFAULT_MATMUL_PRECISION=highest` did not. Why Triton gets it wrong (lost precision or a miscompiled kernel) is NOT known, and no upstream report has been filed.
>
> **Does it affect this sampler?** The sampler uses log-densities and gradients only; it computes no Hessian. On that model the gradients were right: batched over 1,024 chains, as the sampler evaluates them, they matched the CPU to 3e-9 relative, with and without Triton. That was checked for one model, not proven in general, so for a new model compare a few gradients on the GPU against the CPU (`jax.jit(..., backend="cpu")`).
>
> **What to do.** Set the flag before JAX starts, in the shell or in the environment (pixi: `[activation.env]`):
>
> ```
> XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" python my_script.py
> ```
>
> On that model it cost nothing measurable (a 1,024-chain gradient: 76.3 ms with Triton, 76.3 to 76.5 ms without). Anything you build on top of the sampler that uses second derivatives (a Laplace approximation, a Newton optimiser, a Hessian-based metric) must either run with this flag or compute the Hessian on the CPU.

## Why ChEES-HMC, and why many short chains

NUTS adapts the trajectory length of every transition by branching, which is hard to run in lock step on a GPU. ChEES-HMC [1] instead tunes the step size and one trajectory length during warmup (ChEES, the Change in the Estimator of the Expected Square, is the criterion it maximises), then runs plain HMC with those values. Every chain does the same work at each step, so thousands of chains cost little more than a few. ChEES adapts the step size and trajectory length only, not the metric; this package takes the metric from Pathfinder. It calls BlackJAX's `chees_adaptation.base()`, not the public `chees_adaptation(...)` builder.

The sampler runs these steps:
1. Start points from PyMC's `adapt_diag` initialisation, repaired where the log-density or its gradient is not finite.
2. Pathfinder (with jitter) for better starts and a mass matrix.
3. A short search for the initial step size.
4. ChEES warmup: step size, trajectory length and starts.
5. Dynamic HMC with the frozen step size, the trajectory length, and the Pathfinder metric.

## Install

Python 3.11+ (the floor of BlackJAX 1.7). The repository ships a [pixi](https://pixi.sh) workspace that pins the GPU stack:

```
pixi install          # solve and create ./.pixi/envs/default
pixi run gpu-check    # prints the JAX version and the devices
```

With pip: `pip install .` gives CPU JAX; `pip install ".[cuda13]"` installs `jax[cuda13]` (driver >= 580, SM 7.5+ GPU). BlackJAX >= 1.7 is required (the sampler uses its internals). `pixi.toml` keeps numpy below 2.5 and numba at 0.66.0.

## Quick start

```python
import pymc as pm
from pfcheeshmc import sample_pf_chees_hmc

with pm.Model() as model:
    ...                                   # priors and likelihood
    idata = sample_pf_chees_hmc(draws=10, chains=1024, random_seed=42)
```

`idata` is a normal `arviz.InferenceData`. Pass `random_seed`: without it the seed is the current date. Stage options are forwarded as dicts: `pathfinder_kwargs`, `step_size_tune_kwargs`, `chees_kwargs`, `dynamic_hmc_kwargs`; a key a stage does not read raises a `TypeError` that names it. `progressbar=False` silences the bars. The notebooks in `examples/` are `8 Schools example.ipynb`, `GEV Port Pirie example.ipynb` (with a NUTS comparison), `New data every call.ipynb` (one model structure, new data and priors on every call) and `float32.ipynb`.

**Output.** `posterior` is in problem space, including `pm.Deterministic` variables. `sample_stats` has exactly `lp`, `diverging`, `acceptance_rate`, `is_accepted`, `energy` and `num_integration_steps`; `lp` is the log-density of the sampled state, as PyMC's JAX samplers emit it. `idata.attrs` records the resolved configuration, `frozen_step_size` and `frozen_L_nominal` (`tests/test_inference_data.py` pins the contract).

## Choosing chains, draws and warmup

With many short chains (about 1,024 chains and 10 draws) split R-hat measures the estimator, not the sampler, so the acceptance test here is **bias against a long NUTS run**: `b2_max < 0.01`, where `b2` is the squared difference of the ensemble mean and the reference mean over the reference variance, for `x` and `x^2` in the unconstrained space (`benchmarks/README.md`, "The R1 gate"). The warmup length is `chees_kwargs={"draws": W}` (default 2,000); the number of chains and of draws are the arguments `chains` and `draws`.

At 1,024 chains and 10 draws these warmups pass on seeds 1 to 3 (RTX A2000, 2026-10-23):

| model | W | `b2_max`, seeds 1 / 2 / 3 |
|---|---|---|
| 8 Schools | 25 | 0.0007 / 0.0007 / 0.0018 |
| GEV (Port Pirie) | 25 | 0.0014 / 0.0003 / 0.0002 |
| logistic regression, 1000 x 25 | 50 | 0.0039 / 0.0045 / 0.0011 |
| radon, non-centered | 300 | 0.0032 / 0.0025 / 0.0031 |

Command, one model (`gpu_run.sh` is the GPU lock; on the CPU drop it and set `JAX_PLATFORMS=cpu`): `benchmarks/gpu_run.sh pixi run python benchmarks/bias.py --models 8schools --chains 1024 --draws 10 --warmups 25 --seeds 1,2,3`. To scan warmups for your own model, pass `--warmups 10,25,50`; a model of your own needs its own NUTS reference (`benchmarks/make_reference.py`).

## Performance

One call of the sampler against one call of PyMC NUTS with 32 chains on 32 cores, each at settings that pass the gate above on three seeds (NUTS: 200 tuning steps, and 250, 100, 250 and 800 draws per chain for the four models). GPU: RTX A2000 6 GB, 1,024 chains, 10 draws, calls 2 and 3 of one process (nothing compiles; new data costs the same). Both sides 2026-10-23.

| model | GPU, one call | CPU NUTS, one call | ratio |
|---|---|---|---|
| 8 Schools | 0.23 s | 3.42 s | 15x |
| GEV | 0.55 s | 5.16 s | 9.4x |
| logistic 1000 x 25 | 1.43 s | 9.68 s | 6.8x |
| radon, non-centered | 4.8 to 7.4 s | 18.3 s | 2.5 to 3.8x |

Commands. GPU: `benchmarks/gpu_run.sh pixi run python benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3` (W from the table above). CPU NUTS: `pixi run python benchmarks/nuts_cpu.py --models MODEL --tune 200 --draws N --seeds 1,2,3 --out ROWS.jsonl`, the `wall_s` of one `pm.sample` call. Radon's GPU time is the range of four warm calls: it depends on the trajectory length ChEES settles on for the seed. In a **fresh process** the first call matters: with the disk cache warm (next section) the first GPU call took 3.6 s (8 Schools), 6.2 s (GEV), 4.9 s (logistic) and 9.6 s (radon) on 2026-10-24, with the command under "Disk cache" on the GPU and `--new-data 100`, against 4.2, 5.9, 10.0 and 18.9 s for the first NUTS call of a fresh process whose PyTensor C cache is warm (2026-10-23, the first row of `nuts_cpu.py`).

## Compile once, and the disk cache

A call compiles its XLA programs the first time it meets a model structure, a set of array shapes, a chain count and sampler settings, and nothing after that in the same Python process. A new seed, new data, new prior values, a longer or shorter warmup or another number of draws, at the same shapes, compile nothing: data, priors, the metric, the step size and the trajectory length are arguments of the compiled programs. Rebuilding the model object for each dataset is fine; a rebuilt model is recognised as the same structure. One exception: a new prior value that equals another constant of the model (for example `pm.Normal("b", 5.0, 3.0)` next to `pm.Normal("a", 0.0, 5.0)`) is a new structure, and that call compiles once (about 4 s on the CPU); other values compile nothing. A new shape, chain count or model structure compiles once (design: `docs/design.md`; test: `tests/test_compile_once.py`).

`benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3` prints, per call, the wall time and the XLA programs and PyTensor functions compiled. CPU (Xeon Platinum 8280L, `OMP_NUM_THREADS=4`, load below 1, 2026-10-28); calls 2 and 3 compile 0 and 0:

| model | W | first call | call 2 / call 3 |
|---|---|---|---|
| 8 Schools | 25 | 10.2 s | 0.29 / 0.31 s |
| GEV | 25 | 12.3 s | 1.33 / 1.71 s |
| logistic 1000 x 25 | 50 | 11.3 s | 4.46 / 4.41 s |
| radon, non-centered | 300 | 43.4 s | 33.6 / 23.2 s (varies with the trajectory length) |

**The disk cache is on by default.** The first call in a new Python process compiles everything again unless JAX can read the programs from disk. The cached programs hold no data and no prior values, so a new dataset at the same shapes reads the old one's entries. Fresh processes, CPU, 8 Schools, 1,024 chains, 10 draws, W=25, same host and date, running `benchmarks/warm_stages.py . 8schools 1024 10 25 --calls 1 --cache-dir DIR` twice with an empty `DIR`: 10.8 s writing the cache, then 2.8 s reading it (123 of 123 hits; 1.4 MB, 122 files). Without a cache the first call was 10.2 s (table above). On the GPU, run the same command under `gpu_run.sh` with `PFCHEESHMC_COMPILATION_CACHE=1 XDG_CACHE_HOME=DIR`, no `--cache-dir`, and `--new-data 0` the first time, `--new-data 100` the second.

- **Where:** `pfcheeshmc.default_compilation_cache_dir()`, `${XDG_CACHE_HOME:-~/.cache}/pfcheeshmc/jax`, created mode 0700. It is set at the start of each `sample_pf_chees_hmc` call, not at import and not by the class API. If the directory belongs to someone else, is group- or world-writable, or cannot be created, the cache stays off with a `RuntimeWarning`. A compilation cache is trusted code: do not point it at a directory others can write to.
- **It is process-wide.** After the first sampler call every JAX program compiled in the process, yours or another library's, is cached there.
- **Your choice wins.** If a JAX cache directory is set (`JAX_COMPILATION_CACHE_DIR`, `jax.config`, `set_cache_dir`) or `jax_enable_compilation_cache` is off, no JAX setting is touched. Turn the default off with `PFCHEESHMC_COMPILATION_CACHE=0`, read at the call. `pfcheeshmc.enable_compilation_cache(path)` picks a directory and also sets JAX's two thresholds to 0 s and no minimum size, because JAX otherwise writes only programs that took over a second.
- **Size and cleaning:** no limit is set (with JAX's `jax_compilation_cache_max_size` JAX scans the whole directory on every write). The directory grows by one model structure's programs per set of shapes and chain count, not per dataset. It is safe to delete at any time. If a process was killed while writing, JAX warns `Error reading persistent compilation cache entry` and compiles instead.
- **Measuring:** `benchmarks/warm_stages.py` and `benchmarks/bias.py` turn the default off (unless `PFCHEESHMC_COMPILATION_CACHE` is set) so that a first call is cold.

## Other metric sources

`pathfinder_kwargs={"imm_source": ...}` chooses how the mass matrix is built. The default, `mc`, is the covariance of the Pathfinder draws, and it is the one the bias gate above covers. The alternatives are `pf_closed` (the closed-form Pathfinder covariance), `lbfgs_lowrank` (the L-BFGS factors Pathfinder already fitted) and `fisher_lowrank` (a low-rank metric from the gradients at Pathfinder draws, BlackJAX's `fisher_score_low_rank`). All hand the kernel a posterior covariance. No performance claim is made for them. If you try one, watch `num_integration_steps` in `sample_stats`: a jump from tens of steps to hundreds means the metric does not suit the model.

## float32 on the GPU: opt-in, and check the result

float32 is **off by default and is not a safe replacement for float64.** It is much faster on some models and fails on others, and the failure can be silent. Use it only after comparing it with float64 on your model.

**Switch it on** with PyTensor's flag, set before PyTensor is imported (in the shell, or in Python before `import pymc`):

```
PYTENSOR_FLAGS="floatX=float32,blas__ldflags=-llapack -lblas -lcblas -lm" python my_script.py
```

The `blas__ldflags` part is for a checkout whose path contains a space; the benchmark scripts add it themselves. PyTensor's JAX backend then turns `jax_enable_x64` off. **Do not enable `jax_enable_x64` yourself**: the log-density becomes float64 at a float32 position, L-BFGS can fail with "cond branches must have equal output types", and the sampler warns ("floatX is float32 but jax_enable_x64 is on", a `RuntimeWarning`). A notebook that starts with `jax.config.update("jax_enable_x64", True)` must drop that line.

**Measured speed-ups**, one call at 1,024 chains and 10 draws, calls 2 and 3 of one process, RTX A2000, 2026-10-24:

| model | W | float64 | float32 | ratio |
|---|---|---|---|---|
| logistic 1000 x 25 | 50 | 1.39 s | 0.18 s | 7.7x |
| radon, non-centered | 300 | 6.42 s | 2.06 s | 3.1x |

Command, once with and once without the `PYTENSOR_FLAGS` above: `benchmarks/gpu_run.sh pixi run python benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3`. On these two models the bias gate passes on seeds 1 to 3 with step size and trajectory length like float64's (same date and card, the `bias.py` command under "Choosing chains", with the flag).

**Where it fails** (same `bias.py` command, with the flag):
- **GEV, `--models gev --warmups 25`:** the step size collapses. The frozen step size was 3e-5 to 9e-3 on seeds 1 to 3 (float64: about 0.5) and max R-hat 57 to 112. The chains barely move from their Pathfinder starts, so `b2_max` can pass although the sampler is not sampling. The warm call is slower than float64 (0.75 s against 0.47 s).
- **8 Schools, `--warmups 25 --seeds 1,2,...,13`:** 2 of 13 seeds failed the gate with float32 (seeds 1 and 4, `b2_max` 0.014 and 0.019, trajectories of 14 to 15 steps, max R-hat 2.4), against 0 of 13 with float64.

**Checks to run on every float32 result:**
1. The frozen step size (`idata.attrs["frozen_step_size"]`): a value orders of magnitude smaller than float64 gives for the same model is the GEV failure.
2. R-hat on every parameter, against a float64 run of the same settings. With 10 draws per chain R-hat is above 1.01 even in float64 (1.5 to 1.9 on 8 Schools and GEV); a collapsed step size gives values far above that (57 to 112 on GEV).
3. `num_integration_steps` in `sample_stats`, against a float64 run. On GEV the collapsed step size came with hundreds of steps per trajectory on the CPU and about 1 on the GPU (float64: about 7); either way the chains barely move.

Every notebook in `examples/` opens with a `USE_FLOAT32` switch; `examples/float32.ipynb` shows where it helps and what the GEV failure looks like.

## Common issues

1. The sampler is not converging: raise the warmup (`chees_kwargs={"draws": 2_000}`), and compare with `benchmarks/bias.py` on a model that has a NUTS reference.
   If the chains stay near their starts (R-hat far above the 1.5 to 1.9 of 10-draw chains, `lp` in `sample_stats` well below the log-density near the mode), check Pathfinder first: the starts and the metric both come from its one L-BFGS path, and BlackJAX's defaults (30 iterations, `ftol=1e-5`) can stop it far from the mode on a stiff posterior. Raise them with `pathfinder_kwargs={"pf_maxiter": 1000, "pf_ftol": 1e-12}` (also `pf_gtol`, `pf_maxcor`, `pf_maxls`, `pf_elbo_samples`, BlackJAX's `gtol`, `maxcor`, `maxls`, `num_samples`). Measured on a 14-parameter upper-bounded GEV regression (κ ≈ 6e6): the defaults stopped L-BFGS about 130 log-density units below the mode and the chains never left; 1,000 iterations reached it. A longer warmup cannot fix this, because the metric is frozen after Pathfinder.
2. On a shared GPU JAX pre-allocates most of the memory at import. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the pixi environment does).
3. On the GPU, divergence counts can differ by one between identical runs at the same seed: GPU reductions are not bitwise reproducible. On the CPU, runs at the same seed are identical (`tests/test_golden.py`).

## Testing and benchmarks

```
pixi run python -m pytest tests -q
```

The suite runs in one process on the CPU: 8 min 40 s on a loaded shared host (542 tests). `tests/test_golden.py` pins the draws of the default path on the ladder models. `benchmarks/README.md` lists the harness and its exact commands: the bias gate, references, warm-stage timing and CPU NUTS.

## References

[1] M. Hoffman, A. Radul, P. Sountsov, "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo", AISTATS, PMLR 130:3907-3915, 2021. https://proceedings.mlr.press/v130/hoffman21a.html

Maintainer notes: `docs/design.md`. Upstream bug drafts, not filed: `docs/upstream-drafts.md`.

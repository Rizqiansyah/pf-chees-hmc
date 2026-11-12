# Options

Every option `sample_pf_chees_hmc` takes, with its default. **Status**: *default* is what runs when you pass nothing;
*opt-in* is off until you set it and changes the run when you do; *experimental* is opt-in with measurements on few
models; *instrumentation* is kept for measurement, not for use. A key a stage does not read raises a `TypeError` that
names it, before the first stage runs. What a run resolved is recorded in `idata.attrs["config"]` (opt-in keys only when
set, so a default run's record is unchanged).

## Arguments

| argument | default | what it does |
|---|---|---|
| `model` | the `with` context | the PyMC model |
| `draws` | 1,000 | draws per chain after the warmup |
| `chains` | 32 | chains, run in lock step; about 1,024 suits a GPU ([Choosing settings](choosing-settings.md)) |
| `random_seed` | today's date, `YYYYMMDD` | the sampler's seed; each stage derives its own from it |
| `progressbar` | `True` | the warmup and sampling bars (the compiled programs are the same either way) |
| `keep_untransformed` | `False` | also return the draws in the unconstrained space |
| `pathfinder_kwargs`, `step_size_tune_kwargs`, `chees_kwargs`, `dynamic_hmc_kwargs` | `None` | the stage options below |

## Starts and metric: `pathfinder_kwargs`

| key | default | status | what it does |
|---|---|---|---|
| `draws` | 4,000 | default | Pathfinder draws per chain; the pool of starts is `chains x draws` rows before the cap |
| `pf_pool_cap` | `max(4 chains, 4 d^2)` with `imm_source="mc"`, else `4 chains` | default | rows kept in the pool; `0` keeps all; an integer sets the count |
| `init_source` | `"pf"` | default | starts from the Pathfinder pool; `"adapt_diag"`: PyMC's starts |
| `imm_source` | `"mc"` | default | the metric: the pool's covariance. `"pf_closed"`, `"lbfgs_lowrank"`, `"fisher_lowrank"`: opt-in ([Metrics](metrics.md)) |
| `ridge` | 1e-6 | default | added to the `"mc"` covariance's diagonal |
| `fisher_draws`, `max_rank`, `fisher_gamma`, `fisher_cutoff` | 4,096, 10, 1e-5, 2.0 | with `"fisher_lowrank"` | draws with gradients used; directions beyond the diagonal; regularisation; eigenvalues within `[1/cutoff, cutoff]` set to 1 |
| `pf_maxiter`, `pf_ftol`, `pf_gtol`, `pf_maxcor`, `pf_maxls`, `pf_elbo_samples` | 30, 1e-5, 1e-8, 10, 1,000, 200 | default (BlackJAX's) | Pathfinder's L-BFGS path and its ELBO draws; raise the first two on stiff posteriors ([Metrics](metrics.md#pathfinders-l-bfgs-settings)) |
| `pm_init` | `"adapt_diag"` | default | the first starts: the model's initial point; `"jitter+adapt_diag"` adds a U(-1, 1) jitter; another PyMC `init` goes to `pm.init_nuts` |
| `pm_chains` | 1 | default | PyMC start points built (at most `chains`), shared cyclically among the chains |
| `pm_tune` | 500 | default | `tune` for `pm.init_nuts`, which reads it only for `"jitter+adapt_diag_grad"` |
| `pm_seed` | the stage seed | default | seed of PyMC's initialisation |
| `pf_jitter_eps`, `pf_max_backtracks`, `pf_max_restarts`, `pf_restart_radius` | 1e-8, 16, 64, 0.05 | default | start repair: a start without a finite log-density and gradient is jittered, moved back towards the origin, or restarted near it ([Design notes](design.md#4-start-repair)) |
| `score_chunk` | 1,024 | default | rows per batch when scoring the pool |
| `seed`, `chains` | from the arguments | default | the stage's seed and chains |

## Initial step size: `step_size_tune_kwargs`

| key | default | status | what it does |
|---|---|---|---|
| `eps0_probes` | 32 | default | starts the step-size search runs from; the initial step size is the geometric median of their results |
| `seed`, `chains` | from the arguments | default | |

## Warmup: `chees_kwargs`

| key | default | status | what it does |
|---|---|---|---|
| `draws` | 2,000 | default | warmup steps |
| `target_accept` | 0.651 | default | the step size's target for the harmonic mean of the chains' acceptance |
| `adam_lr` | 0.05 | default | Adam's learning rate for ChEES's trajectory length |
| `max_leapfrog_steps` | 1,000 | default | cap on the trajectory length during the warmup, in steps of the current step size; with a fixed `trajectory_length`, also on the draws' steps |
| `metric_updates` | `None` | opt-in | warmup fractions at which the metric is re-estimated from the chains, e.g. `(0.25, 0.5, 0.75)` ([Metrics](metrics.md#metric-updates-during-the-warmup-opt-in)) |
| `metric_shrinkage` | `5 / (chains + 5)` | with `metric_updates` | weight of the old metric in each re-estimate |
| `metric_estimator` | `"cov"` | with `metric_updates`; `"fisher"`, `"fisher_lowrank"` experimental | the re-estimate: the chains' covariance, or the Fisher-divergence metric from positions and gradients |
| `metric_max_rank` | 10 | with `"fisher_lowrank"` | directions beyond the diagonal |
| `acceptance_floor` | `None` | experimental | a floor in (0, `target_accept`) on each chain's acceptance in the step size's statistic, e.g. 0.01 ([Choosing settings](choosing-settings.md#run-to-run-spread-a-step-size-floor-and-a-fixed-trajectory-length-opt-in-experimental)) |
| `trajectory_length` | `None` | experimental | a fixed trajectory length T (integration time) in place of ChEES's ([Choosing settings](choosing-settings.md#run-to-run-spread-a-step-size-floor-and-a-fixed-trajectory-length-opt-in-experimental)) |
| `warmup_stop` | `None` | instrumentation | end the warmup early when a stop rule holds: `{"block": 250, "window": 250, "rhat": 1.01, "eps_stability": 0.5}` (defaults shown; `"accept_band": (lo, hi)` adds a band on the harmonic acceptance). Cost more than it saved on the four benchmark models; cannot be combined with `metric_updates` |
| `progressbar`, `seed`, `chains` | from the arguments | default | |

## Draws: `dynamic_hmc_kwargs`

| key | default | status | what it does |
|---|---|---|---|
| `delayed_rejection` | `None` | opt-in | retry a divergent transition at a smaller step: an integer `a >= 2` (one retry at step / a), or `{"factor": a, "stages": k}` (up to k - 1 retries) ([Choosing settings](choosing-settings.md#divergences-in-a-rare-region-delayed-rejection-opt-in)) |
| `delayed_rejection["retry"]` | `"divergent"` | default; the others experimental | which rejections are retried: `"divergent"`, `{"energy_above": c}` or `"probabilistic"` ([Retry rules](choosing-settings.md#retry-rules-experimental)) |
| `leapfrog_unroll` | 1 | opt-in | leapfrog steps per loop pass; the same draws to rounding ([Performance](performance.md#several-leapfrog-steps-per-loop-pass-opt-in)) |
| `draws`, `progressbar`, `seed`, `chains` | from the arguments | default | |

## Environment variables

| variable | effect |
|---|---|
| `PFCHEESHMC_COMPILATION_CACHE=0` (or `false`, `no`, `off`) | the default on-disk compilation cache stays off ([Compile once](compile-and-cache.md)) |
| `XDG_CACHE_HOME` | where the default cache goes: `$XDG_CACHE_HOME/pfcheeshmc/jax`, else `~/.cache/pfcheeshmc/jax` |
| `JAX_COMPILATION_CACHE_DIR` | a JAX cache of your own; the package then leaves JAX's cache settings alone |
| `PYTENSOR_FLAGS=floatX=float32` | a float32 run, set before PyTensor is imported ([float32](float32.md)) |
| `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` | correct compiled Hessians on the GPU ([GPU notes](gpu-notes.md)) |

## What a run records

`idata.attrs`: `config` (JSON: what each stage resolved, the seeds, and the opt-in settings in use), `diagnostics` (JSON:
Pathfinder's L-BFGS iterations against its cap, divergences by energy; [Troubleshooting](troubleshooting.md#divergences)),
`frozen_step_size`, `frozen_L_nominal` (steps per trajectory before the jitter, `ceil(T / step size)`),
`frozen_inverse_mass_matrix` and its shape, `warmup_steps_used`, the gradient counts `warmup_grad_evals_per_chain`,
`sampling_grad_evals_per_chain` and `sampling_retry_grad_evals_per_chain`, and `pfcheeshmc_version`.

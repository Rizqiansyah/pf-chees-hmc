# Metrics (mass matrices)

## Sources

`pathfinder_kwargs={"imm_source": ...}` chooses how the mass matrix is built.

| `imm_source` | what it is |
|---|---|
| `mc` (default) | the covariance of the Pathfinder draws; the one the [bias gate](choosing-settings.md) covers |
| `pf_closed` | the closed-form Pathfinder covariance |
| `lbfgs_lowrank` | the L-BFGS factors Pathfinder already fitted |
| `fisher_lowrank` | a low-rank metric from the gradients at Pathfinder draws (BlackJAX's `fisher_score_low_rank`) |

All hand the kernel a posterior covariance. No performance claim is made for the alternatives. If you try one, watch
`num_integration_steps` in `sample_stats`: a jump from tens of steps to hundreds means the metric does not suit the model.

`fisher_lowrank` leaves out Pathfinder draws whose gradient is not finite (a draw beyond a hard edge of the support, say),
with a `RuntimeWarning`; the counts are in `idata.attrs["config"]` (`fisher_rows`, `fisher_rows_used`).

## Pathfinder's L-BFGS settings

The starts and the metric both come from one L-BFGS path. BlackJAX's defaults (30 iterations, `ftol=1e-5`) can stop far
from the mode on a stiff posterior. Raise them with

```python
pathfinder_kwargs={"pf_maxiter": 1000, "pf_ftol": 1e-12}
```

Also available: `pf_gtol`, `pf_maxcor`, `pf_maxls`, `pf_elbo_samples` (BlackJAX's `gtol`, `maxcor`, `maxls`,
`num_samples`; defaults 1e-8, 10, 1000, 200). Measured on a 14-parameter upper-bounded GEV regression (condition number
about 6e6): the defaults stopped L-BFGS about 130 log-density units below the mode and the chains never left; 1,000
iterations reached it.

## Metric updates during the warmup (opt-in)

By default the metric is frozen after Pathfinder, before any chain has moved.

```python
chees_kwargs={"metric_updates": (0.25, 0.5, 0.75)}
```

re-estimates it at those fractions of the warmup from the current chain positions:

- it becomes the covariance over chains (dense or diagonal, like the Pathfinder metric), shrunk towards the old metric by
  `metric_shrinkage` (default `5 / (chains + 5)`);
- the step size then restarts from a fresh step-size search under the new metric; the trajectory length carries on;
- an estimate that is not finite and positive definite is not used;
- the updates are recorded in `idata.attrs["config"]` (`metric_updates`, `metric_update_log`);
- they cannot be combined with `warmup_stop`, and add no compiled program after the first call.

Use them when one coordinate mixes far more slowly than the others: Pathfinder under-read its spread, and a frozen metric
keeps it slow for the whole run. Measured on a 12-coordinate upper-bounded GEV regression (1,024 chains, warmup 200, 100
draws, one seed): the `mc` metric alone gave worst R-hat 2.7 and min bulk ESS 1,293. With updates at 0.25/0.5/0.75 it gave
R-hat 1.015 and ESS 69,085, and every coordinate was decorrelated by lag 5. Divergences rose from 2 to 13, and the frozen
step size fell from 0.11 to 0.037, because the step size then answers to a stiff region of that model rather than to the
metric. The bias gate does not cover this option.

### The estimator at an update: covariance or Fisher (opt-in)

```python
chees_kwargs={"metric_updates": (0.25, 0.5, 0.75), "metric_estimator": "fisher_lowrank"}
```

`metric_estimator` chooses what an update estimates:

- `"cov"` (the default): the covariance of the chains' positions, as above;
- `"fisher"`: per coordinate `sqrt(Var[x] / Var[grad log p])`, from the chains' positions and their log-density
  gradients (the warmup state carries them, so no gradient is added). This is the diagonal metric that minimises the Fisher
  divergence between the target and a Gaussian (Seyboldt, Carlson & Carpenter 2026). A dense Pathfinder metric gets it
  as a diagonal matrix, so correlations are dropped;
- `"fisher_lowrank"`: that diagonal plus up to `metric_max_rank` directions (default 10) where the positions and the
  gradients disagree with it, as BlackJAX's `fisher_score_low_rank` estimates them (nutpie's settings: directions whose
  eigenvalue lies in [1/2, 2] are dropped). It needs a dense metric, and is handed to the kernel as a dense matrix.

For a Gaussian target the Fisher estimates are exact from any set of chain positions that spans it (the diagonal one
for a diagonal covariance, the low-rank one when every correlated direction is outside [1/2, 2]); the covariance
estimate has the Monte Carlo error of the number of chains. Chains with a non-finite position or gradient are left out.
Shrinkage, the guard and the step-size restart are those of `"cov"`; a non-default estimator is recorded in the config
(`metric_estimator`, `metric_max_rank`).

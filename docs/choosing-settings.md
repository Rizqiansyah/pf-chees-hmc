# Choosing chains, draws and warmup

The number of chains and of draws are the arguments `chains` and `draws`. The warmup length is
`chees_kwargs={"draws": W}` (default 2,000).

## The acceptance test: bias against a long NUTS run

With many short chains (about 1,024 chains and 10 draws) split R-hat measures the estimator, not the sampler. The
acceptance test used here is therefore **bias against a long NUTS run**: `b2_max < 0.01`, where `b2` is the squared
difference of the ensemble mean and the reference mean over the reference variance, for `x` and `x^2` in the unconstrained
space (`benchmarks/README.md`, "The R1 gate").

At 1,024 chains and 10 draws these warmups pass on seeds 1 to 3 (RTX A2000, 2026-10-23):

| model | W | `b2_max`, seeds 1 / 2 / 3 |
|---|---|---|
| 8 Schools | 25 | 0.0007 / 0.0007 / 0.0018 |
| GEV (Port Pirie) | 25 | 0.0014 / 0.0003 / 0.0002 |
| logistic regression, 1000 x 25 | 50 | 0.0039 / 0.0045 / 0.0011 |
| radon, non-centered | 300 | 0.0032 / 0.0025 / 0.0031 |

Command, one model (`gpu_run.sh` is the GPU lock; on the CPU drop it and set `JAX_PLATFORMS=cpu`):

```bash
benchmarks/gpu_run.sh pixi run python benchmarks/bias.py --models 8schools --chains 1024 --draws 10 --warmups 25 --seeds 1,2,3
```

To scan warmups for your own model, pass `--warmups 10,25,50`. A model of your own needs its own NUTS reference
(`benchmarks/make_reference.py`).

## Long runs: R-hat and ESS

When you want the usual summaries (R-hat below 1.01, a bulk and tail ESS target, 0 divergences) rather than a bias
check against a reference, run more draws per chain, so that each chain is long enough for split R-hat to mean something.
Check `diverging` in `sample_stats`, and the R-hat and ESS of every parameter, as for any MCMC run. A worked example,
1,024 chains of 1,000 draws on a stiff regression, is in the case study (coming in 0.3.1).

## Warmup options

`chees_kwargs` keys (defaults in brackets): `draws` (2,000), `adam_lr` (0.05), `target_accept` (0.651, BlackJAX's ChEES
optimum), `max_leapfrog_steps` (1,000), `metric_updates` and `metric_shrinkage` ([Metrics](metrics.md)), `warmup_stop`,
`progressbar`, `seed`, `chains`.

`warmup_stop` (for example `{"block": 250, "window": 100, "rhat": 1.01}`) ends the warmup early when a stop rule is met.
It is off by default. On all four benchmark models it cost more time at the block boundaries than it saved, so it is kept
for instrumentation only, and it cannot be combined with `metric_updates`.

`step_size_tune_kwargs` takes `eps0_probes` (32): how many starts the initial step-size search uses.

## Divergences in a rare region: delayed rejection (opt-in)

`dynamic_hmc_kwargs={"delayed_rejection": a}` (an integer `a >= 2`; off by default) retries a divergent transition once
from the same point, with the step divided by `a` and `a` times the steps (the same trajectory length). The retry is
accepted with the delayed-rejection probability of Modi, Barnett & Carpenter (2023, *Bayesian Analysis*, eq. 30, with
the retry probability set to 1 on a divergence and 0 otherwise), so the sampler stays exact: a retry is accepted only
if the reverse first proposal from its end point would also have diverged.

Use it after you have diagnosed the divergences, for a region the tuned step was never meant for (for example a hard
edge of the support that the posterior reaches rarely). It is not a substitute for a reparameterisation: divergences
spread over the bulk mean the geometry or the model needs fixing. The run reports both sides:

- `diverging` stays the first-stage divergence, as without the option;
- `retried`, `retry_accepted`, `diverging_after_retry` and `retry_integration_steps` in `sample_stats`, and
  `sampling_retry_grad_evals_per_chain` in the attributes, say what the retries did and cost;
- a `RuntimeWarning` when more than 0.1% of transitions are retried.

Cost: an iteration in which any chain retries runs `(a + 1) n` extra leapfrog steps for every chain (chains move in
lockstep on the GPU); other iterations cost nothing extra.

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
from the same point, with the step divided by `a` and `a` times the steps (the same trajectory length).
`{"delayed_rejection": {"factor": a, "stages": k}}` allows up to `k - 1` retries, at step `/ a`, `/ a^2`, ..., each made
only if the previous one diverged (eq. 29 of the paper; a stage-`j` acceptance needs the reverse proposals of all
earlier stages from its end point, the "ghosts", so stage 3 costs `a^2 + a + 2` trajectories for the chains that reach
it). On a hard-edge test with frequent divergences, 3 stages left 294 transitions divergent where 2 left 11,234. The retry is
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

Cost: `(a + 1) n` leapfrog steps per retried chain, `n` the iteration's step count. Only the chains that need a retry
run it, eight at a time (`compiled.DR_BATCH`), so an iteration without retries costs nothing extra and one with `r`
retries costs `ceil(r / 8)` batches. On a GPU the time follows the leapfrog steps run one after another, not the
gradient count: a step with any retry takes about `a + 1` trajectories longer, whatever the number of chains
retried. Example (1,024 chains, `a = 4`, retries on 10% of steps): gradients +0.06%, sampling time +30%, and the
retry branch adds about 20 s of compilation. `a = 2` is cheaper but recovered fewer divergences there (6 of 101
accepted against 36).

Later stages cost more per chain that reaches them: the proposal at stage `k` and the ghosts its acceptance needs take
`a + 1`, `a^2 + a + 2` and `a^3 + a^2 + 2a + 4` trajectories at stages 2, 3 and 4 (9, 74 and 596 at `a = 8`). They run
only where the stage before diverged, so in practice they are rare. Each stage enlarges the compiled sampling program
once; it is then kept in the [disk cache](compile-and-cache.md). Measured on an upper-bounded GEV regression in float32
on a GPU (1,024 chains x 1,000 draws): `{"factor": 8, "stages": 4}` took 25 to 123 first-stage divergences per run to
none left, for at most 0.2% more gradients; the first run compiled for about 100 s in all.

Scope and checks:

- Only the sampling phase retries. Warmup divergences are as without the option (warmup draws are not kept).
- By default a retry is made only after a divergence; a merely rejected proposal is not retried. Other retry rules
  are below.
- The retry's acceptance test has its own random numbers (`compiled.DR_STREAM`), separate from the first stage's.
- Tested for exactness with two, three and four stages in float64, two and three in float32
  (`tests/test_delayed_rejection.py`: chains started at exact draws of a target with a hard edge keep its law; dropping
  the ghost terms breaks it).

### Retry rules (experimental)

`{"delayed_rejection": {"factor": a, "stages": k, "retry": rule}}` chooses when a later proposal is made after a
rejection:

- `"divergent"` (the default): after a divergence (energy error above 1,000), as above;
- `{"energy_above": c}`: after a rejected proposal whose energy error was above `c` (c > 0). The same indicator family at
  a lower threshold: rejections with a large energy error are retried too, not only divergences;
- `"probabilistic"`: after a rejection, with probability `1 - alpha`, alpha the rejected proposal's acceptance
  probability (Modi, Barnett & Carpenter 2023, s.3.2, eq. 31). A proposal rejected by chance at a high acceptance
  probability is rarely retried; one rejected strongly almost always.

Every rule keeps the sampler exact: the retry probability enters the acceptance at the current state and at the ghosts
(eq. 30). The rule is recorded in the config (`delayed_rejection.retry`) unless it is the default. The 0.1% warning
applies to the default rule only, since the others retry ordinary rejections by design.

These rules retry far more transitions than the default, and on a GPU every iteration with any retry waits for the
retried chains' `(a + 1) n` extra steps. They pay where ordinary rejections, not divergences, are what holds the sampler
back (Neal's funnel: the neck), and mostly on a CPU, where the cost follows the gradients. Tests
(`tests/test_dr_retry_rules.py`): two and three stages keep the law of a hard-edge target and of Neal's funnel under both
rules; dropping the ghost terms breaks it under both; dropping the threshold rule's factor at the ghosts breaks it on the
funnel. Dropping the probabilistic rule's factor moved the test statistic too little to detect at 200,000 chains (the
factor matters little on these targets), so its exactness rests on eq. (30) and the invariance tests.

# Case study: a stiff extreme-value regression

pf-chees-hmc on one workstation GPU against PyMC's NUTS on a 56-core CPU, on five datasets of the same model.

**Headline.** At the same quality bar (no divergences, R-hat below 1.01) and the same effective sample size,
pf-chees-hmc on an RTX A2000 was **19 to 76x faster than NUTS using all 56 cores** and **250 to 1,080x faster than
NUTS with the usual 4 chains** (the range covers five datasets and two to six pf-chees-hmc seeds each). The posterior
quantiles agreed to within 0.3%; on one dataset the upper bound differed by a small but consistent amount, up to 0.21%
(see [Agreement](#agreement)).

## The model

A regression for block maxima: a generalised extreme value (GEV) likelihood whose mean and upper bound are smooth
functions of a covariate, each a polynomial with a change point (two change points, both sampled), the upper bound
with a Beta-marginal Gaussian-copula prior at five reference points, and one shape parameter, negative (a bounded
upper tail). 14 free parameters; five datasets (A to E) of 1,500 to 1,900 observations in 14 groups. The data sit
close to the upper bound in some groups (a hard support edge) and the change points have long, thin tails. NUTS needed
step sizes of 0.004 to 0.011 and 370 to 880 leapfrog steps per draw (whole-CPU runs). The model and data are not public; both samplers
ran the identical PyMC model, in the same parametrisation.

## Set-up

| | pf-chees-hmc | PyMC NUTS, 4 chains | PyMC NUTS, whole CPU |
|---|---|---|---|
| hardware | NVIDIA RTX A2000 (6 GB) | 4 cores of 2 x Xeon Platinum 8280L | 56 cores (2 x 28) of the same |
| chains x draws | 1,024 x 1,000 | 4 x 5,000 | 56 x 28,300 to 45,500 (sized to reach the pf-chees-hmc ESS) |
| warmup / tuning | Pathfinder (L-BFGS: 1,000 iterations, ftol 1e-12; Fisher low-rank metric), then 200 ChEES steps with the metric re-estimated at 25, 50 and 75% | 1,000 steps (diagonal metric, PyMC's default) | 1,000 steps |
| step size and trajectory | ChEES, target acceptance 0.8, then frozen | NUTS, target acceptance 0.99, maximum tree depth 10 | same |
| starts | Pathfinder draws | `init="adapt_diag"` (see [What it took](#what-it-took)) | same |
| software | pf-chees-hmc 0.3, JAX, BlackJAX 1.7.1 | PyMC 5.28.5 | same |

Time is the wall time of one sampler call, from the model to the `InferenceData`: tuning or warmup, the draws, and
for pf-chees-hmc compilation (with the [disk cache](compile-and-cache.md) as it was, so part of it may have been read
from disk). The pf-chees-hmc runs and the whole-CPU NUTS runs were one dataset at a time with nothing else heavy
running; the five 4-chain NUTS runs ran side by side, 4 cores each (20 of 56 cores in use).

Effective sample size (ESS) is the smallest bulk or tail ESS over every parameter and every reported quantity;
R-hat is the worst rank-normalised split R-hat over the same.

## Results

pf-chees-hmc ran with seeds 1 and 2 on every dataset, and seeds 3 to 6 on dataset E; NUTS ran once per set-up.

| dataset | sampler | wall time | min ESS | ESS per second | worst R-hat | divergences |
|---|---|---|---|---|---|---|
| A | pf-chees-hmc, seeds 1 / 2 | 288 / 456 s | 570,465 / 461,249 | 1,978 / 1,012 | 1.0015 / 1.0017 | 0 |
| | NUTS, 56 chains | 12,915 s | 682,488 | 52.8 | 1.0001 | 0 |
| | NUTS, 4 chains | 1,785 s | 7,170 | 4.0 | 1.0004 | 0 |
| B | pf-chees-hmc, seeds 1 / 2 | 565 / 516 s | 299,908 / 354,407 | 531 / 687 | 1.0014 / 1.0013 | 0 |
| | NUTS, 56 chains | 20,640 s | 391,310 | 19.0 | 1.0001 | 0 |
| | NUTS, 4 chains | 3,242 s | 4,381 | 1.4 | 1.0012 | 0 |
| C | pf-chees-hmc, seeds 1 / 2 | 547 / 787 s | 270,242 / 421,198 | 494 / 535 | 1.0029 / 1.0015 | 0 |
| | NUTS, 56 chains | 22,670 s | 326,154 | 14.4 | 1.0001 | 0 |
| | NUTS, 4 chains | 3,844 s | 4,107 | 1.1 | 1.0017 | 0 |
| D | pf-chees-hmc, seeds 1 / 2 | 162 / 168 s | 647,248 / 630,740 | 3,992 / 3,743 | 1.0014 / 1.0014 | 0 |
| | NUTS, 56 chains | 16,438 s | 864,739 | 52.6 | 1.0000 | 0 |
| | NUTS, 4 chains | 1,656 s | 6,098 | 3.7 | 1.0006 | 0 |
| E | pf-chees-hmc, seeds 1 to 6 | 334 to 616 s | 352,900 to 409,676 | 658 to 1,056 | 1.0014 to 1.0020 | 0 |
| | NUTS, 56 chains | 27,336 s | 477,068 | 17.5 | 1.0001 | 0 |
| | NUTS, 4 chains | 2,953 s | 3,858 | 1.3 | 1.0006 | 0 |

The pf-chees-hmc wall time varies from seed to seed with the trajectory length ChEES settles on (on A, 288 s at a
nominal 23 leapfrog steps for seed 1 and 456 s at 41 for seed 2); ESS per second varies less, but still by up to a
factor of 2.

Speed-up, as the ratio of ESS per second (the time NUTS would need to reach the pf-chees-hmc ESS, over the pf-chees-hmc
time, if its ESS grows in proportion to its draws), over the pf-chees-hmc seeds:

| dataset | A | B | C | D | E |
|---|---|---|---|---|---|
| against NUTS, 56 chains | 19 to 37x | 28 to 36x | 34 to 37x | 71 to 76x | 38 to 61x |
| against NUTS, 4 chains | 250 to 490x | 390 to 510x | 460 to 500x | 1,020 to 1,080x | 500 to 810x |

The whole-CPU runs check that proportionality: they did reach the pf-chees-hmc ESS, and their ESS per second was 13.2
to 14.3 times that of the 4-chain runs, close to the 14 times the cores.

(agreement)=
## Agreement

Compared: the 5%, 50% and 95% posterior quantiles of the 31 quantities a user of the model reports (the mean and the
upper bound in each of the 14 groups, the shape, the two change points), 93 per dataset, as z = difference / combined
Monte Carlo standard error. For two runs of the same posterior about 5% of |z| exceed 2 and almost none exceed 3; the
quantiles are correlated (the upper bounds of neighbouring groups move together), so a single fluctuation can push
several of them past 2 at once.

| | largest \|z\| per dataset (A to E) | quantiles with \|z\| > 3 | largest relative difference |
|---|---|---|---|
| two pf-chees-hmc seeds | 3.06, 2.59, 2.31, 2.22, 2.81 | 1 of 465 | 0.18% |
| NUTS, 4 chains, against pf-chees-hmc | 2.70, 2.39, 2.72, 2.26, 1.90 | 0 of 465 | 0.85% |
| NUTS, 56 chains, against pf-chees-hmc | 2.08, 2.63, 2.70, 2.10, 2.33 | 0 of 465 | 0.28% |

On four datasets NUTS and pf-chees-hmc differ no more than two pf-chees-hmc seeds do. On dataset E they differ
slightly but consistently in the upper bound: NUTS (56 chains) puts it higher against each of six pf-chees-hmc seeds
(42 of 42 upper-bound quantiles higher against the mean of seeds 1 and 2; against the mean of all six, 21% of |z| above
2, none above 3, the largest relative difference 0.21%). The six pf-chees-hmc seeds agree with each other as runs of
one posterior should, and their spread matches their reported Monte Carlo error. The same direction, weaker, shows on
dataset C. C and E are the two datasets on which NUTS most often stopped at its maximum tree depth (15% and 26% of
draws). Which sampler is off is not known: NUTS is a comparison here, not the truth, and a difference of 0.2% is far
below what matters for this model, but it is larger than Monte Carlo error.

## What it took

- NUTS with PyMC's defaults failed on this model. The jittered starts of `init="jitter+adapt_diag"` had log densities
  of -1e40 to -1e85; from there the step size fell to 1e-22 within 20 tuning steps, and 7 of 8 chains never reached
  the posterior (R-hat 6.7). Starting every chain at the model's initial point (`init="adapt_diag"`) with target
  acceptance 0.99 fixed it. Even so, on the two hardest datasets (C and E) 12 to 28% of NUTS draws stopped at the
  maximum tree depth: an efficiency warning, not a bias.
- pf-chees-hmc needed Pathfinder's L-BFGS settings raised (`pf_maxiter=1000`, `pf_ftol=1e-12`: BlackJAX's defaults
  stopped far from the mode) and the warmup metric updates (`metric_updates=[0.25, 0.5, 0.75]`); see
  [Metrics](metrics.md).
- Both samplers ran the same reparametrised model: sampler coordinates in which the support edge and the change-point
  tails are easier to integrate (an exact change of variables, so the same posterior). With the model's original
  coordinates pf-chees-hmc had a few divergences per run; NUTS was not run on them.

## Caveats

- Hardware and algorithm are compared together: a mid-range workstation GPU against a dual-socket server CPU.
- One model. On the four small benchmark models in [Performance](performance.md) the GPU speed-up per call is 2.5 to
  15x; a stiff posterior, which forces NUTS into small steps and long trajectories, is where many short chains gain most.
- One NUTS run per set-up and dataset; two pf-chees-hmc seeds per dataset (six on E), whose wall times differed by up
  to 58%.
- ESS and its ratio depend on the quantity; the minimum over all of them is used throughout.

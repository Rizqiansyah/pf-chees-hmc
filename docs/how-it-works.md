# How it works

## Why ChEES-HMC, and why many short chains

NUTS adapts the trajectory length of every transition by branching, which is hard to run in lock step on a GPU. ChEES-HMC
[^chees] instead tunes the step size and one trajectory length during warmup (ChEES, the Change in the Estimator of the
Expected Square, is the criterion it maximises), then runs plain HMC with those values. Every chain does the same work at
each step, so thousands of chains cost little more than a few.

ChEES adapts the step size and trajectory length only, not the metric; this package takes the metric from Pathfinder. The
warmup adapter is BlackJAX's `chees_adaptation.base`, called as it is; the package records a per-step trace (step size,
trajectory length, harmonic-mean acceptance) beside its state for the progress bar and diagnostics.

## The pipeline

1. **Starts.** PyMC's `adapt_diag` initialisation, repaired where the log-density or its gradient is not finite.
2. **Pathfinder** (with jitter): better starts and a mass matrix (the metric).
3. **Step size.** A short search for the initial step size.
4. **ChEES warmup** (default 2,000 steps): step size, trajectory length and starts. Optionally the metric is re-estimated
   from the chains at given fractions of the warmup ([metric updates](metrics.md#metric-updates-during-the-warmup-opt-in)),
   the step size's statistic gets a floor, and the trajectory length is fixed instead of adapted
   ([run-to-run spread](choosing-settings.md#run-to-run-spread-a-step-size-floor-and-a-fixed-trajectory-length-opt-in-experimental)).
5. **Draws.** HMC with the frozen step size, the trajectory length and the metric (Pathfinder's, or the last update's).
6. **Transform** back to problem space, including deterministics.

Every stage is a compiled JAX program whose data and prior values are arguments, so a new seed, new data or new priors at
the same shapes compile nothing ([Compile once](compile-and-cache.md)). Maintainers' details: [Design notes](design.md).

[^chees]: M. Hoffman, A. Radul, P. Sountsov, "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte
    Carlo", AISTATS, PMLR 130:3907-3915, 2021. <https://proceedings.mlr.press/v130/hoffman21a.html>

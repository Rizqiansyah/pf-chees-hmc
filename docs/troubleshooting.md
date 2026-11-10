# Troubleshooting

## The sampler is not converging

1. Raise the warmup (`chees_kwargs={"draws": 2_000}`), and compare with `benchmarks/bias.py` on a model that has a NUTS
   reference.
2. **If the chains stay near their starts** (R-hat far above the 1.5 to 1.9 of 10-draw chains, `lp` in `sample_stats` well
   below the log-density near the mode), check Pathfinder first. The starts and the metric both come from its one L-BFGS
   path, and BlackJAX's defaults can stop it far from the mode on a stiff posterior. The sampler warns when L-BFGS used
   every iteration it was given; `idata.attrs["diagnostics"]` has the count (`pathfinder.lbfgs_iterations` against
   `pathfinder.lbfgs_maxiter`). Raise
   `pathfinder_kwargs={"pf_maxiter": 1000, "pf_ftol": 1e-12}` ([Metrics](metrics.md#pathfinders-l-bfgs-settings)). A longer
   warmup cannot fix this, because the metric is frozen after Pathfinder (unless `metric_updates` is set, and even then the
   starts still come from Pathfinder).
3. **If one coordinate mixes far more slowly than the others**, try
   [metric updates](metrics.md#metric-updates-during-the-warmup-opt-in).

## Divergences

Read `diverging` in `sample_stats`. The frozen step size and metric are in `idata.attrs`, so a divergent transition can be
replayed with the sampler's own settings. On the GPU, make the run bit-identical first
([GPU notes](gpu-notes.md#runs-at-the-same-seed-can-differ-on-the-gpu)).

Then read `energy` at the divergent draws. `idata.attrs["diagnostics"]` has the counts (`divergence_energy`: `inf`,
`nan`, `-inf`, `finite`), and the sampler warns when any divergence ended at a NaN or `-inf` energy:

- **`+inf`**: the proposal left the support (the log density is `-inf` there). If these are rare and sit at a hard edge
  the posterior reaches only now and then, [delayed rejection](choosing-settings.md#divergences-in-a-rare-region-delayed-rejection-opt-in)
  retries them with a smaller step and keeps the sampler exact.
- **`NaN`** (or `-inf`, a log density of `+inf`): the log density or its gradient failed to evaluate. Often that is the
  model's arithmetic (an overflow, a function evaluated where it loses all digits, float32 rounding), which no step size
  fixes; find the term first.
- **finite but large**: the step is too large for the curvature there.

Divergences spread over the bulk mean the geometry or the model needs fixing, whichever kind they are.

## A `TypeError` naming an option

A stage was given a key it does not read (or one that was removed). The message lists the valid keys.

## The GPU runs out of memory, or other users do

[Shared GPUs](gpu-notes.md#shared-gpus).

## The first call in every new process is slow

That is the compile. Check that the [disk cache](compile-and-cache.md#the-disk-cache-is-on-by-default) is on, and that its
directory is yours and not group- or world-writable (otherwise it stays off with a `RuntimeWarning`).

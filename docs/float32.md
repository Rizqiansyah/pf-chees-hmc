# float32 on the GPU: opt-in, and check the result

float32 is **off by default and is not a safe replacement for float64.** It is much faster on some models and fails on
others, and the failure can be silent. Use it only after comparing it with float64 on your model.

## Switching it on

Use PyTensor's flag, set before PyTensor is imported (in the shell, or in Python before `import pymc`):

```bash
PYTENSOR_FLAGS="floatX=float32,blas__ldflags=-llapack -lblas -lcblas -lm" python my_script.py
```

The `blas__ldflags` part is for a checkout whose path contains a space; the benchmark scripts add it themselves. PyTensor's
JAX backend then turns `jax_enable_x64` off.

:::{important}
Do not enable `jax_enable_x64` yourself. The log-density becomes float64 at a float32 position, L-BFGS can fail with "cond
branches must have equal output types", and the sampler warns ("floatX is float32 but jax_enable_x64 is on", a
`RuntimeWarning`). A notebook that starts with `jax.config.update("jax_enable_x64", True)` must drop that line.
:::

## Measured speed-ups

One call at 1,024 chains and 10 draws, calls 2 and 3 of one process, RTX A2000, 2026-10-24:

| model | W | float64 | float32 | ratio |
|---|---|---|---|---|
| logistic 1000 x 25 | 50 | 1.39 s | 0.18 s | 7.7x |
| radon, non-centered | 300 | 6.42 s | 2.06 s | 3.1x |

Command, once with and once without the `PYTENSOR_FLAGS` above:
`benchmarks/gpu_run.sh pixi run python benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3`. On these two models the bias
gate passes on seeds 1 to 3 with step size and trajectory length like float64's (same date and card, the `bias.py` command
under [Choosing chains, draws and warmup](choosing-settings.md), with the flag).

## Where it fails

Same `bias.py` command, with the flag:

- **GEV, `--models gev --warmups 25`:** the step size collapses. The frozen step size was 3e-5 to 9e-3 on seeds 1 to 3
  (float64: about 0.5) and max R-hat 57 to 112. The chains barely move from their Pathfinder starts, so `b2_max` can pass
  although the sampler is not sampling. The warm call is slower than float64 (0.75 s against 0.47 s).
- **8 Schools, `--warmups 25 --seeds 1,2,...,13`:** 2 of 13 seeds failed the gate with float32 (seeds 1 and 4, `b2_max`
  0.014 and 0.019, trajectories of 14 to 15 steps, max R-hat 2.4), against 0 of 13 with float64.

## Checks to run on every float32 result

1. The frozen step size (`idata.attrs["frozen_step_size"]`): a value orders of magnitude smaller than float64 gives for the
   same model is the GEV failure.
2. R-hat on every parameter, against a float64 run of the same settings. With 10 draws per chain R-hat is above 1.01 even
   in float64 (1.5 to 1.9 on 8 Schools and GEV); a collapsed step size gives values far above that (57 to 112 on GEV).
3. `num_integration_steps` in `sample_stats`, against a float64 run. On GEV the collapsed step size came with hundreds of
   steps per trajectory on the CPU and about 1 on the GPU (float64: about 7); either way the chains barely move.

`examples/float32.ipynb` shows where it helps and what the GEV failure looks like.

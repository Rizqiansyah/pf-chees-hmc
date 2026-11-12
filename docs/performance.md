# Performance

## One call against PyMC NUTS on 32 cores

One call of the sampler against one call of PyMC NUTS with 32 chains on 32 cores, each at settings that pass the
[bias gate](choosing-settings.md) on three seeds (NUTS: 200 tuning steps, and 250, 100, 250 and 800 draws per chain for the
four models). GPU: RTX A2000 6 GB, 1,024 chains, 10 draws, calls 2 and 3 of one process (nothing compiles; new data costs
the same). Both sides measured 2026-10-23.

| model | GPU, one call | CPU NUTS, one call | ratio |
|---|---|---|---|
| 8 Schools | 0.23 s | 3.42 s | 15x |
| GEV | 0.55 s | 5.16 s | 9.4x |
| logistic 1000 x 25 | 1.43 s | 9.68 s | 6.8x |
| radon, non-centered | 4.8 to 7.4 s | 18.3 s | 2.5 to 3.8x |

Radon's GPU time is the range of four warm calls: it depends on the trajectory length ChEES settles on for the seed.

Commands:

```bash
# GPU (W from the table in "Choosing chains, draws and warmup")
benchmarks/gpu_run.sh pixi run python benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3
# CPU NUTS; the wall_s of one pm.sample call
pixi run python benchmarks/nuts_cpu.py --models MODEL --tune 200 --draws N --seeds 1,2,3 --out ROWS.jsonl
```

## The first call in a fresh process

With the [disk cache](compile-and-cache.md) warm, the first GPU call took 3.6 s (8 Schools), 6.2 s (GEV), 4.9 s (logistic)
and 9.6 s (radon) on 2026-10-24, with `--new-data 100`. The first NUTS call of a fresh process whose PyTensor C cache is
warm took 4.2, 5.9, 10.0 and 18.9 s (2026-10-23, the first row of `nuts_cpu.py`).

## Loops inside the log density

On a GPU, every pass of a `while` loop inside the compiled program costs a round trip to the CPU: XLA copies the loop
condition to the host to decide whether to go on. The sampler's own loops pay this once per leapfrog step. A log density
that contains a loop pays it once per pass, inside every gradient. Iterative special functions are the usual source: JAX's
`betainc` (behind PyTensor's Beta and Student-t CDFs, for example) runs a continued fraction for up to 200 passes in
float32. On one model whose likelihood called it, each leapfrog step took 2.5 times as long as with a version of the same
continued fraction with a fixed number of passes (same accuracy).

To check a model, look for loops in its compiled gradient:

```python
txt = jax.jit(jax.grad(logp)).lower(x).compile().as_text()
[line for line in txt.splitlines() if " while(" in line]
```

A loop whose line carries `known_trip_count` has a fixed number of passes and needs no copy.

## Several leapfrog steps per loop pass (opt-in)

```python
dynamic_hmc_kwargs={"leapfrog_unroll": 4}
```

runs the sampling phase's leapfrog steps four per pass of the trajectory loop (then the remainder one by one), so the loop
condition goes to the host about a quarter as often. The steps are the same; the draws agree with the default to rounding
(XLA may fuse an unrolled pass differently), with the same accept decisions on the CPU. Delayed rejection's retries are
unrolled too. Measured on a GPU (an upper-bounded GEV regression, float32, 1,024 chains, 300 draws from one warmed-up
state, three repeats, spread under 1%):

| steps per pass | sampling time, two models | extra compile time, no retries | with 4-stage delayed rejection |
|---|---|---|---|
| 2 | -3.9% / -6.0% | +7.5 s | |
| 4 | -7.4% / -7.3% | +15 s | -7.8%, +81 s compile |
| 8 | -7.1% / -7.8% | +31 s | |

The compiled program grows with the unrolling (once per loop it contains), and the [disk cache](compile-and-cache.md)
keeps it, so the time is paid once per model shape. Use 4 for long runs on a GPU; on a CPU the loop test costs nothing.

## When these numbers were taken

The tables were measured on 2026-10-23 and 2026-10-24, before 0.3.0. 0.3.0 leaves the draws at default settings unchanged
(the CPU golden tests), but the timings have not been measured again.

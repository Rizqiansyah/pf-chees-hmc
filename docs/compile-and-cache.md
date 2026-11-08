# Compile once, and the disk cache

## Compile once per structure

A call compiles its XLA programs the first time it meets a model structure, a set of array shapes, a chain count and
sampler settings, and nothing after that in the same Python process.

- A new seed, new data, new prior values, a longer or shorter warmup or another number of draws, at the same shapes,
  compile nothing: data, priors, the metric, the step size and the trajectory length are arguments of the compiled programs.
- Rebuilding the model object for each dataset is fine; a rebuilt model is recognised as the same structure.
- One exception: a new prior value that equals another constant of the model (for example `pm.Normal("b", 5.0, 3.0)` next
  to `pm.Normal("a", 0.0, 5.0)`) is a new structure, and that call compiles once (about 4 s on the CPU).
- A new shape, chain count or model structure compiles once.

Design: [Design notes](design.md); test: `tests/test_compile_once.py`.

`benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3` prints, per call, the wall time and the XLA programs and PyTensor
functions compiled. CPU (Xeon Platinum 8280L, `OMP_NUM_THREADS=4`, load below 1, 2026-10-28); calls 2 and 3 compile 0 and 0:

| model | W | first call | call 2 / call 3 |
|---|---|---|---|
| 8 Schools | 25 | 10.2 s | 0.29 / 0.31 s |
| GEV | 25 | 12.3 s | 1.33 / 1.71 s |
| logistic 1000 x 25 | 50 | 11.3 s | 4.46 / 4.41 s |
| radon, non-centered | 300 | 43.4 s | 33.6 / 23.2 s (varies with the trajectory length) |

## The disk cache is on by default

The first call in a new Python process compiles everything again unless JAX can read the programs from disk. The cached
programs hold no data and no prior values, so a new dataset at the same shapes reads the old one's entries.

Measured: fresh processes, CPU, 8 Schools, 1,024 chains, 10 draws, W=25, same host and date, running
`benchmarks/warm_stages.py . 8schools 1024 10 25 --calls 1 --cache-dir DIR` twice with an empty `DIR`: 10.8 s writing the
cache, then 2.8 s reading it (123 of 123 hits; 1.4 MB, 122 files). Without a cache the first call was 10.2 s. On the GPU,
run the same command under `gpu_run.sh` with `PFCHEESHMC_COMPILATION_CACHE=1 XDG_CACHE_HOME=DIR`, no `--cache-dir`, and
`--new-data 0` the first time, `--new-data 100` the second.

Where
: `pfcheeshmc.default_compilation_cache_dir()`, `${XDG_CACHE_HOME:-~/.cache}/pfcheeshmc/jax`, created mode 0700. It is set
  at the start of each `sample_pf_chees_hmc` call, not at import and not by the class API. If the directory belongs to
  someone else, is group- or world-writable, or cannot be created, the cache stays off with a `RuntimeWarning`. A
  compilation cache is trusted code: do not point it at a directory others can write to.

It is process-wide
: After the first sampler call every JAX program compiled in the process, yours or another library's, is cached there.

Your choice wins
: If a JAX cache directory is set (`JAX_COMPILATION_CACHE_DIR`, `jax.config`, `set_cache_dir`) or
  `jax_enable_compilation_cache` is off, no JAX setting is touched. Turn the default off with
  `PFCHEESHMC_COMPILATION_CACHE=0`, read at the call. `pfcheeshmc.enable_compilation_cache(path)` picks a directory and also
  sets JAX's two thresholds to 0 s and no minimum size, because JAX otherwise writes only programs that took over a second.

Size and cleaning
: No limit is set (with JAX's `jax_compilation_cache_max_size` JAX scans the whole directory on every write). The directory
  grows by one model structure's programs per set of shapes and chain count, not per dataset. It is safe to delete at any
  time. If a process was killed while writing, JAX warns `Error reading persistent compilation cache entry` and compiles
  instead.

Measuring
: `benchmarks/warm_stages.py` and `benchmarks/bias.py` turn the default off (unless `PFCHEESHMC_COMPILATION_CACHE` is set)
  so that a first call is cold.

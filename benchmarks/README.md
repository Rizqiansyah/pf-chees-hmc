# benchmarks

The small harness that judges a change to the sampler: does it still sample the right distribution (bias against a long NUTS
run), how long does a call take stage by stage, and how does CPU NUTS compare. Nothing here is imported by the package, and
nothing here imports from `tests/`.

| file | what it is for |
|---|---|
| `models.py` | the four ladder models in Python: `8schools`, `gev`, `logit_1000x25`, `radon_noncentered`. `build_model(name)` gives a fresh PyMC model. |
| `bias.py` | **the R1 gate.** Runs `sample_pf_chees_hmc` at chains C, draws D, warmup W and a seed, and scores `b2_max` against the NUTS reference. One cell, one JSON row. |
| `references.json` | the reference manifest: for each model the sha256 of the reference draws, and the settings and the gate numbers they were made with. |
| `make_reference.py` | rebuilds a reference (a long PyMC NUTS run) and updates its manifest entry. |
| `warm_stages.py` | time per stage, XLA compiles and PyTensor function compiles of repeated calls: the cost of a first call, a warm call, new data, the disk cache. |
| `nuts_cpu.py` | CPU NUTS (32 chains on 32 cores) scored by the same instrument as the gate, for the time-to-pass comparison. |
| `gpu_run.sh` | the GPU lock: every GPU process on the shared card runs under one `flock`. |
| `compile_counter.py`, `seeded_models.py` | counters for XLA and PyTensor compiles, and models that are rebuilt with new data or prior values at fixed shapes. `warm_stages.py` and the tests use them. |

## Environment

From a checkout, with the pixi environment. A CPU run, and what every command below assumes:

```bash
cd /path/to/pf-chees-hmc            # the path may contain a space; the scripts handle it
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=4
```

For the GPU leave the first two unset (the pixi environment sets `CUDA_VISIBLE_DEVICES=0`) and put `benchmarks/gpu_run.sh` in
front of the command (below). Each script sets what a path with a space needs (`blas__ldflags` in `PYTENSOR_FLAGS`, added to
yours, not instead of it) and turns the on-disk compilation cache off unless `PFCHEESHMC_COMPILATION_CACHE` is set, so that
the first call of a process is cold. floatX is PyTensor's: `PYTENSOR_FLAGS=floatX=float32` before the command (float32 is an
opt-in; the gate shows what it costs).

The reference draws are outside git, in `~/pf-runs/references/<model>.npz` (or `$PF_REFERENCE_DIR`). `bias.py` refuses a file
whose sha256 is not the manifest's:

```bash
pixi run python benchmarks/bias.py --models 8schools,gev,logit_1000x25,radon_noncentered --check-references-only
# [ref-guard] 8schools: PASS  sha256 matches the manifest, 3 value vars, 10 leaves, 80000 reference draws   (and so on)
```

## The R1 gate: `bias.py`

In the many-short-chain regime (C about 1,024, D about 10) split R-hat measures the estimator, not the sampler, so the gate is
bias against the reference, in the unconstrained space: `b2(f) = (mean_ens f - mean_ref f)^2 / var_ref f` for `f` in
`{x_i, x_i^2}`, and `b2_max` is its maximum. `b2_floor` is its own Monte-Carlo noise. The verdict `g_r1` is `pass` when `C*D >= 2,000`, the floor is at most 0.005 and
`b2_max < 0.01`. It is `fail` when `b2_max >= 0.01` at such a floor, or when, at a coarser floor, `b2_max` is above both 0.01
and three floors. `not resolvable` (a coarse floor, a `b2_max` inside its noise) and `not eligible` (`C*D < 2,000`) are not
passes. A warmup passes when every seed passes.

```bash
# CPU: 8 Schools, C=1024, D=10, W=25, seeds 1-3 (the gate cells for the other models: gev W=25, logit_1000x25 W=50, radon_noncentered W=300)
pixi run python benchmarks/bias.py --models 8schools --chains 1024 --draws 10 --warmups 25 --seeds 1,2,3 --out ~/pf-runs/bias.jsonl

# GPU: the same, under the lock (bias.py refuses a GPU run without it)
benchmarks/gpu_run.sh pixi run python benchmarks/bias.py --models 8schools --chains 1024 --draws 10 --warmups 25 --seeds 1,2,3

# several warmups at once, and options of the sampler (JSON; the row records them)
pixi run python benchmarks/bias.py --models 8schools --warmups 10,25,50 --pathfinder '{"imm_source": "pf_closed"}'
```

Each cell appends one row to `--out` (default `~/pf-runs/bias.jsonl`) and prints one line. The row has the cell (`model`,
`chains`, `draws_per_chain`, `warmup`, `seed`), `b2_max`, `b2_floor`, `b2_argmax` (the leaf and the moment), `g_r1`, `max_rhat`
(information only), `frozen_step_size`, `frozen_L_nominal`, gradients per chain, `divs_per_1k`, `wall_s` (`cold` marks the first
cell of the process, which includes the compile), `effective` (the configuration the sampler resolved), `reference_sha256`, and
`draws_sha256`, the hash of the unconstrained draws: a change that must not move the draws can say so by leaving it equal.

## References: `make_reference.py`

A long PyMC NUTS run (8 chains, 10,000 draws, 2,000 tuning, `target_accept` 0.95; GEV 0.99) on the ladder model, gated at max
R-hat < 1.01 and bulk ESS > 10,000 per parameter. The settings default to the model's entry in `references.json`, so this
rebuilds what the manifest describes, and writes the new sha256 into the manifest (the file is not byte-reproducible):

```bash
pixi run python benchmarks/make_reference.py 8schools          # ~/pf-runs/references/8schools.npz, and references.json
pixi run python benchmarks/make_reference.py gev --out /tmp/gev_check.npz     # elsewhere; the manifest is left alone
```

## Warm-stage timing: `warm_stages.py`

```bash
# TREE MODEL CHAINS DRAWS WARMUP: the first call (cold) and calls 2-3 (warm, a new seed each)
pixi run python benchmarks/warm_stages.py . 8schools 1024 10 25 --calls 3

# the same with new observed data at the same shapes on every call (the "new data every run" workflow)
pixi run python benchmarks/warm_stages.py . 8schools 1024 10 25 --calls 3 --new-data 100

# the on-disk compilation cache: run twice in fresh processes (the first writes it, the second reads it)
pixi run python benchmarks/warm_stages.py . 8schools 1024 10 25 --calls 1 --cache-dir /tmp/pf-cache
pixi run python benchmarks/warm_stages.py . 8schools 1024 10 25 --calls 1 --cache-dir /tmp/pf-cache
```

`TREE` is the tree whose `pfcheeshmc/` is measured: `.`, or a `git archive` of an earlier commit to compare. `MODEL` is a ladder
name, or `hier`, the 8-Schools-shaped model of `seeded_models.py`, rebuilt with new data and prior values on every call. Each
call prints the wall time and, per stage (init, pathfinder, eps0, warmup, sampling, transform, rest), the seconds, the XLA
programs compiled and the PyTensor functions compiled. `--json OUT` keeps them.

## CPU NUTS: `nuts_cpu.py`

32 chains on 32 cores at a given (tune, draws), scored by the instrument of the gate (the same reference, `score_ensemble` and
`g_r1`), with the exact gradient count. One thread per chain, so that cores equal chains; do not run it during a timed GPU batch.
Run each model once untimed first (the PyTensor C-module cache is on disk).

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    pixi run python benchmarks/nuts_cpu.py --models 8schools --tune 200 --draws 250 --seeds 1,2,3 --out ~/pf-runs/nuts_cpu.jsonl
```

`C*D` is `32 * draws`: below 2,000 (draws under 63) the verdict is always `not eligible`.

## Tests

`tests/test_bias.py` and `tests/test_nuts_cpu.py` pin the scoring (the statistic, the floor, the verdict wording, the leaf
flattening, the reference reader and its hash check, the row), `tests/test_reference_npz.py` the shape contract of the reference
files and the manifest, and `tests/test_benchmark_models.py` that the four models are the ones the references were made on.

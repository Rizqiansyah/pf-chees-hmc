# C5+C6 SUMMARY: deletions, design page, upstream drafts, pyproject, README

Branch `cleanup-c56` from `cleanup` b42af4d, head ff245ae. Model: fast-tier. CPU only. Own measurements: Intel Xeon Platinum 8280L (112 threads), `OMP_NUM_THREADS=4`, 2026-10-28.

## DEVIATIONS
1. **"The draws program per `min(D, 1000)`" is not true of the code, so `docs/design.md` does not say it.** No such program exists at HEAD or at the tag. The sampling loop is `compiled.sampling_chunk`, a program of `SAMPLE_CHUNK` (50) steps with a dynamic active count, called `ceil(D/50)` times by `compiled.run_chunks` (`sampler.py:_run_dynamic_hmc`). The transform is `compiled.transform_draws` in chunks of `TRANSFORM_CHUNK` (50) (`sampler.py:_transform_in_draw_chunks`). Warmup is the same (`WARMUP_CHUNK` 50). `design.md` states this chunk layout, with D and W as host loop bounds.
2. **The README's CPU compile-once table and disk-cache numbers are re-measured today, not copied.** The old numbers had a "32-core host" and no date (first call 13.5 / 17.8 / 11.9 / 16.9 s, calls 2-3 0.39 / 1.0 / 3.2 / 4.6 s). Radon's calls 2-3 here are 33.6 and 23.2 s, not 4.6 s: a different host and thread setting, and radon's wall depends on the trajectory length ChEES settles on. The README says so.
3. **`pixi.lock` was rewritten by `pixi lock`** (run in the checkout; it solves and installs nothing). The diff is 2 insertions and 1 deletion, all inside the `pfcheeshmc` editable entry's `requires_dist`. `jax[cuda13]>=0.9.0` became `jax>=0.9.0`, plus `jax[cuda13]>=0.9.0 ; extra == 'cuda13'`. No package entry changed. `pixi list` of the shared env before and after is identical except `pfcheeshmc` 3.33 to 3.34 MiB, which is the editable package's own directory size. The allowed condition holds, so the change is kept. `pixi.toml` is untouched.
4. **`tests/float32_run.py:15` has no stale reference at this head** (C4's line numbers had moved). The remaining `tools/r1_bias` mentions in that file are tag mode (grep below). Also fixed beyond the C4 list, comments and docstrings only: `pfcheeshmc/utils.py:101` (`see docs/ladder-results.md`), `tests/test_metric_orientation.py:183` (`LEDGER entry`), and the README pointers in `sampler.py` and `compilation_cache.py` (the section is now "Compile once, and the disk cache").

## Spec items
| # | item | status | where |
|---|---|---|---|
| C5.1 | delete `results/`, `docs/*` except `docs/cleanup/` and the new pages, the workflow-skill directory; keep `ref/` | DONE | commit d574cdc; `ls docs` is `cleanup design.md upstream-drafts.md`; `ref/README.md` kept |
| C5.2 | `docs/design.md` | DONE (DEV 1) | seven sections, each citing `file:function`: rule C (`lifted.py:_candidates, _build, _widen_model_inputs, RULES`); compile once (`compiled.py`, `lifted.py:_key, _REGISTRY`); lift memo and disk cache (`lifted.py:_MEMO, _recall, _signature`; `compilation_cache.py:ensure_default_compilation_cache`); start repair (`sampler.py:_usable_mask, _ensure_finite_inits, _find_finite_start`); metric convention (`sampler.py:_run_pathfinder, _kernel_permutation, _metric_to_kernel_order`); erfcx (`lifted.py:_band, _erfcx_guarded, jax_funcify_Erfcx`); float32. Each fact was read against the code. |
| C5.3 | `docs/upstream-drafts.md`, both marked not filed | DONE | (a) PyMC Uniform: NaN confirmed on the C backend and on PyMC's JAX backend (pymc 5.28.5, pytensor 2.38.2). Cause confirmed as an int8 wrap: `np.int8(100) - np.int8(-100) == -56`; width 127 is fine, width 128 gives NaN. (b) JAX erfcx: 0 on [26.54326, 26.64175] in float64 and on [9.19455, 9.41928] in float32, confirmed on jax 0.11.2 CPU; gradient -1.128 in the band. The code blocks in the file were extracted and re-run, with identical output. |
| C5.4 | `pyproject.toml` | DONE (DEV 3) | version 0.2.0, `jax>=0.9.0`, extra `cuda13 = ["jax[cuda13]>=0.9.0"]` |
| C5.5 | stale references | DONE (DEV 4) | `compiled.py:323`, `sampler.py:100,104`, `tests/conftest.py:23`, README (rewritten) |
| C5.6 | grep acceptance | DONE | below |
| C6 | README rewrite per the outline | DONE | 145 lines (was 159). `grep -n "pending\|caveat\|pf-r1\|LEDGER\|retract" README.md` finds nothing. |
| C6.a | every number cites a command | DONE | table below. The lift cost "0.05-0.14 s per call" is dropped: `warm_stages.py` prints stage times and compile counts, not the lift cost. |
| C6.b | `imm_source` | DONE | "Other metric sources": option kept, the three alternatives named, no performance claim, `mc` is the one the gate covers |
| C6.c | float32 section | DONE | switch-on (`PYTENSOR_FLAGS` with the blas flag, no `jax_enable_x64`, the warning), 7.7x and 3.1x with commands, GEV collapse and 8 Schools 2 of 13 stated plainly, three checks (step size, R-hat, `num_integration_steps`), pointer to `examples/`. No GEV mitigation proposed. |
| C6.d | Testing section | DONE | suite command, 8 min 40 s on a loaded shared host (C3), pointer to `benchmarks/README.md` |

## Grep result
`grep -rn "results/\|docs/review\|tools/\|r1_bias\|t3_cpu\|\workflow-skill dir" . --exclude-dir=.git --exclude-dir=.pixi`, outside `docs/cleanup/`, leaves five lines. All name the tag's tree: the git-archive mode of two runners that compare with `archive/pre-cleanup`, whose models are built by its own `tools/r1_bias`.
- `tests/make_golden.py:14` (docstring) and `:46` (`from r1_bias import build_model`, in the branch for a tree without `benchmarks/`).
- `tests/float32_run.py:7` (docstring), `:49` (comment "a tree before benchmarks/ (the tag)") and `:51` (`from r1_bias import build_model`, same branch).
Inside `docs/cleanup/` (process records, deleted at C8) the pattern matches by nature.

## README numbers
Hardware and dates as in the README. "tag" means the file is at `archive/pre-cleanup`.
| README number | source | command that produces that kind of row | checked |
|---|---|---|---|
| `b2_max` seeds 1-3, four models at W = 25 / 25 / 50 / 300, C=1024, D=10 (RTX A2000, 2026-10-23) | F2f `results/f2/F2f/REPORT.md` "R1 bias gate" (rows in `results/f2/runs/F2f__default.jsonl`, tag) | `benchmarks/gpu_run.sh pixi run python benchmarks/bias.py --models M --chains 1024 --draws 10 --warmups W --seeds 1,2,3` | ran `bias.py` on the CPU at C=64: `b2_max` 0.04526 = the C4 acceptance value. C4 reproduced the 8 Schools C=1024 row on the CPU to about 1e-14 relative. The other three models at C=1024 were not re-run. |
| GPU one call 0.23 / 0.55 / 1.43 / 4.8-7.4 s | F2f `REPORT.md` Headline (2026-10-23) | `benchmarks/gpu_run.sh pixi run python benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3` | ran on the CPU at C=64 (prints wall, XLA compiles and PyTensor functions per call); GPU not run |
| CPU NUTS 3.42 / 5.16 / 9.68 / 18.3 s; tune 200, draws 250 / 100 / 250 / 800 | F2f `REPORT.md` "R2-T4 redone" (script `cpu_nuts_calls.py`, deleted, timed `pm.sample` only) | `benchmarks/nuts_cpu.py --models MODEL --tune 200 --draws N --seeds 1,2,3`, field `wall_s` | ran at 4 chains, tune 50, draws 50 (row with `wall_s`); not at 32 chains on 32 cores (UNVERIFIED) |
| ratios 15x / 9.4x / 6.8x / 2.5-3.8x | F2f `REPORT.md` Headline | quotient of the two rows above | arithmetic |
| fresh-process first call, disk cache warm, GPU 3.6 / 6.2 / 4.9 / 9.6 s (2026-10-24) | F2g `results/f2/F2g/SUMMARY.md` "GPU re-check" (tag) | `PFCHEESHMC_COMPILATION_CACHE=1 XDG_CACHE_HOME=DIR benchmarks/gpu_run.sh pixi run python benchmarks/warm_stages.py . MODEL 1024 10 W --calls 1 --new-data S`, twice (`--new-data 0`, then `100`) | CPU version run (row below); GPU not run |
| CPU NUTS first call, fresh process, warm C cache 4.2 / 5.9 / 10.0 / 18.9 s (2026-10-23) | F2f `REPORT.md` "Workflow B" | first row of `nuts_cpu.py` in a fresh process after an untimed run | not re-run; same UNVERIFIED as the NUTS row |
| CPU first call 10.2 / 12.3 / 11.3 / 43.4 s; calls 2 / 3: 0.29 / 0.31, 1.33 / 1.71, 4.46 / 4.41, 33.6 / 23.2 s; 0 compiles on calls 2-3 | this task (2026-10-28, load 0.4 at start) | `warm_stages.py . MODEL 1024 10 W --calls 3`, four models, `OMP_NUM_THREADS=4` | measured |
| CPU disk cache: 10.8 s write, 2.8 s read, 123 of 123 hits, 1.4 MB, 122 files; no cache 10.2 s | this task | `warm_stages.py . 8schools 1024 10 25 --calls 1 --cache-dir DIR`, twice with an empty `DIR`; `du -sh DIR` | measured |
| float32 GPU warm 1.39 -> 0.18 s (7.7x), 6.42 -> 2.06 s (3.1x) (2026-10-24) | F2k `results/f2/F2k/gpu.md` and `SUMMARY.md` "GPU (lead)" (tag) | `benchmarks/gpu_run.sh pixi run python benchmarks/warm_stages.py . MODEL 1024 10 W --calls 3`, with and without `PYTENSOR_FLAGS=floatX=float32` | CPU float32 run of the same command (logit, C=64): header printed `floatX: float32 jax_enable_x64: False`, 0 compiles on call 2; GPU not run |
| float32 gate passes on logit and radon, eps and L like float64's | F2k `SUMMARY.md` gate table | `bias.py` as above, with the flag | not run (C=1024) |
| GEV float32: eps 3e-5 to 9e-3, max R-hat 57-112, warm call 0.75 vs 0.47 s; float64 eps about 0.5 | F2k `gpu.md` (eps 2.77e-05 / 7.37e-04 / 8.88e-03; R-hat 112 / 92 / 57; f64 eps 0.542 / 0.519 / 0.507) | `bias.py --models gev --warmups 25` with the flag (row fields `frozen_step_size`, `max_rhat`); warm call from `warm_stages.py` | CPU float32 `bias.py --models gev --chains 64 --draws 10 --warmups 25 --seeds 1` printed `eps=0.00193 L=266 splitRhat=4.45`: the collapse reproduces on the CPU |
| 8 Schools float32 W=25: 2 of 13 seeds fail (seeds 1 and 4, `b2_max` 0.0142 and 0.0191, L 14-15, R-hat 2.4); float64 0 of 13 | F2k `SUMMARY.md` seeds 4-13 table | `bias.py --models 8schools --warmups 25 --seeds 1,...,13`, with and without the flag | not run (13 seeds at C=1024) |
| 8 min 40 s, 542 tests | C3 SUMMARY | `pixi run python -m pytest tests -q` | this task's own run: 542 passed in 7:44 (below) |
| warmup default 2,000, `chees_kwargs={"draws": W}` | `pfcheeshmc/sampler.py:_run_chees` | none (code) | read |

## Full suite
Command: `pixi run --manifest-path "$MAIN/pixi.toml" env PYTHONPATH="$WT" python -m pytest "$WT/tests" -q --rootdir "$WT" -p no:cacheprovider`, CPU, `OMP_NUM_THREADS=4`. Load average 4.63 at the start (1 minute), 11.25 at the end (1 minute, mostly the suite itself). Line, verbatim:
`542 passed, 48 warnings in 464.01s (0:07:44)`
(An earlier launch of the same script printed nothing and ended after 50 s; the cause was not found. The second launch is the one recorded.) The README keeps C3's 8 min 40 s on a loaded host.

## UNVERIFIED
- Whether `nuts_cpu.py`'s `wall_s` at 32 chains on 32 cores reproduces F2f's NUTS walls. Those came from `cpu_nuts_calls.py` (deleted), which timed `pm.sample` only. This host has 112 threads.
- No GPU row of the README was re-run (no GPU). Each command was run on the CPU at a small size.
- `bias.py` at C=1024 for gev, logit_1000x25 and radon_noncentered on the CPU; only 8 Schools was reproduced, by C4.
- The mechanism of the JAX `erfcx` band is attributed in the draft from the matching edges, not from JAX's source. The GPU backend, float16 and bfloat16 are untried for it. The float32 grid probe (5.6 million points, within 1e-5 of SciPy) is F2k's and was not re-run.
- Other PyMC versions and other integer-bound distributions for the Uniform draft.
- Dropped from the README, with no replacement: ESS/s chain-count scaling, the warmup 1,000 / 2,000 / 10,000 table, the `imm_source` ratios, the "lift cost 0.05-0.14 s per call".

Integrator suite on merged `cleanup` (ad40567): `542 passed, 48 warnings in 460.34s (0:07:40)`, rc 0, load 2.79 → 13.39

Integrator step-1 verdicts: scope, no behaviour change (comments only), pixi.lock/pyproject, deletions, run-time reads, dangling refs, design.md citations, DEVIATION 1 (no `min(D,1000)` in code), upstream-drafts reproducers (re-run, match), README numbers and float32 section: OK. Findings: (1) README NUTS command lacked `--out`: fixed by the lead in b95bfbf. (2) "Part B" / "pf-r1-int" labels in comments: fixed in b95bfbf. (3) GPU rows not re-run: carried to C7/C8.

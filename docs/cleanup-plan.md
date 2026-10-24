# Repo cleanup — scope (draft for the owner, 2026-10-24)

**Goal.** A repo that a new user or maintainer can read in an afternoon. It holds:
- the package;
- tests of the package;
- the examples;
- a README that states what is true today;
- one small benchmark harness, used to judge future changes.

The history stays reachable through a git tag, not in the tree. No sampler default changes. Draws on the default
path stay bit-identical.

## Today (local `main` b34e66e, 384 tracked files)
| area | files | what it is | essential? |
|---|---|---|---|
| `pfcheeshmc/` | 8 (225 KB; `sampler.py` 98 KB, 1,806 lines) | the package | yes, but it carries about 40 stage options, many of them experiment arms (below) |
| `tests/` | 45 (34 test modules, 596 tests, 12.5 min) | package tests, tool tests, task proofs | partly |
| `tools/` | 33 | the bias harness (`r1_bias.py`, 52 KB), ladder models, about 25 one-off probes from runs 1–2 and F2 | a few |
| `results/` | 280 | ledgers, SUMMARYs, run rows and drivers of run 1, run 2 and F2 | history |
| `docs/` | 9 (275 KB) | plans and reviews; `ladder-results.md` is built on since-retracted measurements | history |
| `examples/` | 2 notebooks (2 MB with outputs) | 8 Schools, GEV | yes, but stale |
| `README.md` | 159 lines | mixes current facts with "pf-r1 caveat … re-measurement pending" paragraphs and retracted tables | rewrite |
| other | the workflow-skill doc, `ref/README.md` (the PDFs are ignored) | | no / small |

### Couplings that decide the order
1. **The ladder models come from the notebooks.** `tools/bench_ladder.load_notebook_model` runs the notebooks'
   cells, the `jax_enable_x64` cell included. 20 test files (modules and child runners) and every tool depend on that. The notebooks cannot be
   rewritten until the models live in Python.
2. **Six test modules prove refactors against old commits** (`git archive 44a0556`, `62c38a3`, `80f0916` …):
   test_compile_once, test_chunked_loops, test_pm_chains, test_start_repair, test_widen and test_float32. They
   were task evidence. Going forward, one golden-draws fixture made at the tag does the same job without git.
3. **Tool tests** (test_r1_bias, test_summarise_rows, test_t3_cpu, test_reference_npz, test_pool_cap_rows,
   test_bench_ladder_stub_restore and test_nested_rhat) go with whatever happens to the tools.
4. **The NUTS reference draws** the bias gate scores against live outside the repo (`~/pf-runs/references`). Their
   manifest is `results/references/MANIFEST.json`, which must move if `results/` goes.
5. **The upstream bug drafts** (`results/f2/F2a/SUMMARY.md`, plus the float32 `erfcx` band from F2k) must move if
   `results/` goes.

## Proposed end state
```
pfcheeshmc/            package (the API pruned if Q3 says so)
tests/                 package tests only; suite target about 5 min on CPU
  data/golden_*.npz    default-path draws per ladder model at fixed seeds, made at the tag
benchmarks/            README with the exact commands
  models.py            the four ladder models in Python (same graphs as the notebooks build today)
  bias.py              the R1 gate (b²_max against NUTS references), trimmed from tools/r1_bias.py
  references.json      the reference manifest (sha256, seeds, settings); make_reference.py rebuilds them
  make_reference.py, warm_stages.py, nuts_cpu.py, gpu_run.sh
examples/              8 Schools, GEV, and a new "new data every call" notebook (the owner's workflow)
docs/
  design.md            one maintainer page: rule C lifting, compile-once layout, start repair, metric convention
  upstream-drafts.md   the PyMC Uniform int8 and JAX erfcx (float64 and float32 bands) drafts, not filed
README.md, pyproject.toml, pixi.toml, pixi.lock, .gitignore
```
Everything else is deleted from the tree. It stays reachable through the tag `archive/pre-cleanup` at b34e66e
(the tags `pf-r1`, `pf-r2`, `archive/pf-r1-int` and `archive/wp-wp4` already exist).

## Work packages (sequential fast-tier tasks, the same rules as F2; the lead reviews, merges and runs the suite)
| WP | what | acceptance |
|---|---|---|
| **C0** (lead) | tag `archive/pre-cleanup`; branch `cleanup` from `main`; save the golden draws (4 models × 2 seeds, CPU, small shape) | tag and fixture committed |
| **C1** | `benchmarks/models.py`: the four ladder models in Python. Tools and tests switch to it, and the notebook loader and its x64 workaround go | for each model, the same value variables, the same `logp` at 100 points, and the same draws as the notebook-built model |
| **C2** (if Q3 = prune) | remove the experiment options and their code paths (list below) | golden draws bit-identical; compile-once tests pass; an unknown option raises a clear `TypeError` |
| **C3** | tests: drop the git-archive proofs in favour of `tests/test_golden.py`; drop the tests of deleted tools; merge small modules; keep compile counts, the cache, float32, the statistical checks and the output contract | the suite passes in about 5 min; no package line loses coverage it has today (coverage diff in the SUMMARY) |
| **C4** | `benchmarks/` as above: `bias.py` trimmed to what the gate needs; the references manifest moved; a benchmarks README | the R1 gate rows reproduce F2f's b² on 8schools at C=64 on the CPU |
| **C5** (lead + task) | delete `results/`, `docs/*` (except the two new pages); write `docs/design.md` and `docs/upstream-drafts.md`; pyproject: version 0.2.0, `jax[cuda13]` as an extra rather than a hard dependency | no dangling path in the tree (`grep` for `results/`, `docs/review`, `tools/`) |
| **C6** | README rewrite (outline below) | every number cites a reproducible command in `benchmarks/`; no "pending" or "caveat" paragraphs |
| **C7** (task writes, lead runs on the GPU) | notebooks: drop the forced `JAX_PLATFORM_NAME=gpu` and the `jax_enable_x64` cell; current API (`random_seed`, `progressbar`, many short chains); the NUTS comparison kept in GEV; a third notebook for new data every call; executed on the GPU with outputs committed | each runs top to bottom on the GPU and on the CPU |
| **C8** (lead) | full suite, the notebooks on the GPU, merge into local `main` | **not pushed** |

**Q3: experiment options proposed for removal** (default path untouched). The task verifies each against the code
before removing it.
- **Reproduce known bugs:** `imm_orientation="precision"` (the inverted metric) and `imm_order` other than
  `"kernel"` (the permutation).
- **Metric refresh** (`metric_refresh*`, 7 options; run 2 D3 = recommend OFF) and `superchains` / nested R-hat.
- **Single-factor experiment arms never adopted:** `iterate` (D1 = NO), `init_pick="iid"`, `pf_start_scale`,
  `pf_center`.
- **Kept:**
  - the default path's options (`init_source`, `pm_chains`, `pm_*`, `max_leapfrog_steps`, `target_accept`,
    `pf_pool_cap`, `score_chunk`, `eps0_probes`, the start-repair knobs, `warmup_stop`);
  - `imm_source` (`mc`, `lbfgs_lowrank`, `fisher_lowrank`, `pf_closed`): the README offers the low-rank metrics.
    Its numbers need re-measuring on one convention, or the option should be dropped (part of C6).

**README outline (C6):**
- what it is and why many short chains;
- install;
- quick start;
- choosing chains, draws and warmup (the bias-gated W per model, from F2f);
- performance (F2's GPU numbers against CPU NUTS);
- compile once and the disk cache (a third of today's length);
- float32 (per Q4);
- the output contract;
- common issues;
- testing and benchmarks;
- references.

## Cost and risk
- **Cost:** 5–6 tasks of about 1–1.5 h each, plus the lead's GPU runs. That is 1–2 days of wall time at
  the time limit.
- **Main risk:** C1. If a Python-built model differs from the notebook's in one detail (a prior's dtype, the
  name order), the NUTS references no longer fit it. The `logp` and draws equality check is the guard.
- **Nothing outward-facing:** no push, and nothing filed.

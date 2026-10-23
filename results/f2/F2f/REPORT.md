# F2f: measurement of `f2` against `main`, and R2-T4 redone (2026-10-23)

Everything here was measured back to back on one host, 22:00–22:41. The host had load 1–4 from our
own jobs; another user's process used about 2% CPU. The GPU is an RTX A2000 6 GB, and only our jobs
ran on it.

- **Trees:**
  - `main` is a `git archive` of 232ae58;
  - `f2` is 3e0a739, which contains F2a–F2e.
- **Shapes:** M4. C=1,024 chains, D=10 draws, and each model at its M2a smallest passing W: 8schools 25,
  gev 25, logit_1000x25 50, radon_noncentered 300.
- **Raw data:**
  - `results/f2/runs/F2f/*.json`;
  - `results/f2/runs/F2f__default.jsonl` (gate rows);
  - logs in `~/pf-runs/f2f/logs/`.
- **Drivers:** `results/f2/drivers/` (`f2f_gpu.sh`, `run_step.sh` with `F2f_gate.cmds`, `f2f_cpu.sh`).
- **Tables:** `stages.md` (all GPU stage tables) and `gate.md`, from `f2f_table.py` and
  `f2f_gate_table.py`.

## Headline
| | 8schools | gev | logit_1000x25 | radon_noncentered |
|---|---|---|---|---|
| GPU warm call, `main` | 6.03 s | 10.21 s | 8.17 s | 20.9 s |
| GPU warm call, `f2` | **0.23 s** | **0.55 s** | **1.43 s** | **4.8–7.4 s** |
| speed-up | 26× | 19× | 5.7× | 2.8–4.3× |
| CPU NUTS per call (32 chains, M5 passing config) | 3.42 s | 5.16 s | 9.68 s | 18.3 s |
| **GPU `f2` vs CPU NUTS, same accuracy** | **15×** | **9.4×** | **6.8×** | **2.5–3.8×** |
| run 2's R2-T4 (GPU `main` vs CPU NUTS) | 0.56× | 0.51× | 1.10× | 0.77× |

**R1 gate:** `f2` passes all 4 models on all 3 seeds.

**New data every run:** in one process this costs `f2` nothing:

| model | `f2` warm, same data | `f2` warm, new data |
|---|---|---|
| 8schools | 0.23 s | 0.22 s |
| gev | 0.55 s | 0.56 s |
| logit_1000x25 | 1.43 s | 1.50 s |
| radon_noncentered | 6.50 s | 4.87 s |

Calls 2–3 make 0 XLA and 0 PyTensor compiles.

## R1 bias gate on `f2` (`gate.md`)
`tools/r1_bias.py` through the run-2 driver: run 2's M4 default rows, seeds 1–3, 2 reps each.

| model | W | b²_max seeds 1–3, run 2 (main) → f2 | G-R1 |
|---|---|---|---|
| 8schools | 25 | 0.0007 0.0007 0.0018 → **identical** | pass 3/3 |
| gev | 25 | 0.0014 0.0003 0.0002 → **identical** | pass 3/3 |
| logit_1000x25 | 50 | 0.0039 0.0045 0.0011 → **identical** | pass 3/3 |
| radon_noncentered | 300 | 0.0039 0.0034 0.0047 → **0.0032 0.0025 0.0031** | pass 3/3 |

- **8schools, gev, logit.** Every logged quantity equals run 2's: b², ε, L, gradients per chain and
  divergences. The draws are unchanged.
- **Radon** changed by design: the start picks (F2c) and the refresh/stop-rule streams (F2b-2).
  - Its b² is lower on every seed.
  - L is 20/21/15 against run 2's 20/16/15. Seed 2 now lands in the L≈21 mode, which costs 2,608
    against 1,778 gradients per chain.
  - Divergences: 0.

**Peak GPU memory** (process-wide, in the gate rows), main → f2:
- 8schools 224 → 272 MiB;
- gev 288 → 416 MiB;
- logit 582 → 212 MiB;
- radon 1,242 → 1,172 MiB.

All are far inside 6 GB. The radon row also closes the UNVERIFIED item "memory at C=1,024, d=175".

## R2-T4 redone: two workflows
Both sides are at equal accuracy. GPU is the default path at the M4 shapes, which passes G-R1 above.
CPU is PyMC NUTS with 32 chains on 32 cores at run 2's M5 passing config, which passes G-R1 on 3/3
seeds (`results/r2/runs/M5__cpu_nuts.jsonl`):

| model | tune | draws |
|---|---|---|
| 8schools | 200 | 250 |
| gev | 200 | 100 |
| logit_1000x25 | 200 | 250 |
| radon_noncentered | 200 | 800 |

The GPU walls are `tools/warm_stages.py`: the sampler call only, blocking on its result. The CPU walls
are `pm.sample` only (`cpu_nuts_calls.py`).

### Workflow A: repeat runs in one process (a notebook or session), with the same data or new data each run
| model | CPU NUTS per call | GPU `main` warm | GPU `f2` warm | f2 vs CPU |
|---|---|---|---|---|
| 8schools | 3.42 s (3.32–3.67) | 6.03 s | 0.22–0.24 s | **15×** |
| gev | 5.16 s (4.78–5.62) | 10.21 s | 0.55–0.56 s | **9.4×** |
| logit_1000x25 | 9.68 s (8.89–10.32) | 8.17 s | 1.40–1.52 s | **6.8×** |
| radon_noncentered | 18.3 s (17.8–20.2) | 20.9 s | 4.8–7.4 s | **2.5–3.8×** |

- **CPU NUTS gives the same wall with new data as with the same data.** New data median vs
  repeat/rebuilt median:
  - 8schools 3.36 vs 3.47 s;
  - gev 5.12 vs 5.16 s;
  - logit 9.85 vs 9.18 s;
  - radon 19.4 vs 18.2 s.

  PyTensor's C cache does not recompile for new data values. The run-2 addendum assumed CPU would pay
  a compile on new data; it does not, in one process.
- **GPU radon** depends on the seed through the ChEES mode (L≈21 vs 15). 4.8–7.4 s is the range of the
  four warm calls (seeds 2–3, same data and new data). On `f2`, 5.6 of the 6.5 s mean is the W=300
  warmup itself.

### Workflow B: one run per new Python process (scripts, batch jobs): the first call
| model | CPU NUTS, PyTensor cache cold / warm | GPU `main`, no cache / disk cache with same data / disk cache with new data | GPU `f2`, no cache / **disk cache with new data** |
|---|---|---|---|
| 8schools | 13.1 / 4.2 s | 36.8 / 6.4 / 10.7 s | 17.2 / **3.7 s** |
| gev | 22.5 / 5.9 s | 46.7 / 9.0 / 15.2 s | 34.6 / **7.4 s** |
| logit_1000x25 | 18.0 / 10.0 s | 51.3 / 7.7 / 11.5 s | 31.9 / **4.8 s** |
| radon_noncentered | 34.2 / 18.9 s | 81.1 / 25.9 / 24.9 s | 43.4 / **9.8 s** |

- **"Cache warm" means different things on each side.**
  - PyTensor's C cache is on by default and lives on disk, so "warm" is a CPU user's normal state.
  - JAX's disk cache is opt-in (`pfcheeshmc.enable_compilation_cache`, F2e).
  - `main` used JAX's three env vars, because it has no helper.
- **With the disk cache,** `f2`'s first call in a new process beats CPU NUTS on 3 of 4 models: 8schools,
  logit and radon. It loses on gev, 7.4 s against 5.9 s; see the gev init finding below.
- **Without the disk cache,** a new process costs `f2` 17–43 s against CPU NUTS's 4–19 s. Most of that is
  compiling Pathfinder: 9–18 s, mostly `pathfinder_fit`.
- **`main` with a warm disk cache and new data** still misses: 10.7–24.9 s. Its programs have the data
  compiled in as constants.

## Per-stage, warm call (GPU, seconds, mean of calls 2–3, same data; full tables in `stages.md`)
| model | tree | wall | init | pathfinder | ε0 | warmup | sampling | transform | rest |
|---|---|---|---|---|---|---|---|---|---|
| 8schools | main | 6.03 | 0.53 | 1.13 | 1.02 | 1.67 | 1.52 | 0.01 | 0.16 |
| 8schools | f2 | 0.23 | 0.02 | 0.04 | 0.00 | 0.05 | 0.02 | 0.01 | 0.07 |
| gev | main | 10.21 | 1.70 | 1.83 | 1.47 | 2.78 | 2.02 | 0.06 | 0.36 |
| gev | f2 | 0.55 | **0.21** | 0.04 | 0.00 | 0.12 | 0.05 | 0.02 | 0.10 |
| logit_1000x25 | main | 8.17 | 0.49 | 1.37 | 1.15 | 3.15 | 1.76 | 0.00 | 0.24 |
| logit_1000x25 | f2 | 1.43 | 0.01 | 0.07 | 0.01 | 1.09 | 0.19 | 0.02 | 0.05 |
| radon_noncentered | main | 20.90 | 0.92 | 7.98 | 1.82 | 7.24 | 2.48 | 0.03 | 0.43 |
| radon_noncentered | f2 | 6.50 | 0.04 | 0.25 | 0.02 | 5.60 | 0.29 | 0.17 | 0.13 |

The fixed per-call cost (init + Pathfinder + ε0) that run 2 named as the cause was 2.8–10.8 s. It is now
0.06–0.31 s.

## Findings
1. **GEV's init runs the log-density eagerly** (pre-existing; also on `main`: 143 compiles in init).
   - GEV's initial point has a NaN gradient by design (xi = 0). So the one shared start (`pm_chains=1`)
     goes through `_find_finite_start` (`sampler.py:350`). Its `_is_usable` calls `self.ld_fn` and
     `jax.grad(self.ld_fn)` outside any jit.
   - That costs 156 one-off primitive compiles: 4.4 s CPU, about 9 s GPU in a cold call. Every warm call
     then pays the op-by-op dispatch: 0.21 of GEV's 0.55 s.
   - **Universal fix:** evaluate the check with `compiled.usable_mask` on a one-row stack. It is already
     jitted, and already compiled for that shape by `_usable_mask`.
     - The decision and the returned point are unchanged, so the draws should be too.
     - Every model whose start needs repair benefits.
   - A small task (F2g) if the owner wants it.
2. **The harness's `wall_s` includes scoring.** `tools/r1_bias.py:586–660` stamps `wall_s` after
   `score_ensemble` and `az.rhat`.
   - That is about 4 s on radon (175 variables at C=1,024), and about 0.15 s on 8schools.
   - Run 2's R2-T4 GPU walls carried it in both trees; for example, its radon 24.93 s included about
     4 s of scoring.
   - The T4 above uses `warm_stages` walls instead.
   - The gate rows' `wall_s` has the same extra: `f2` radon rep 1 shows 9.3–11.2 s against a stage sum
     of 5.4–7.3 s.
3. **Cold compile is now mostly Pathfinder.**
   - `pathfinder_fit`, plus about 100 tiny eager ops, takes 9–18 s of a 17–45 s cold call on the GPU.
   - The disk cache removes it: a warm-cache first call compiles in 1.0–2.0 s.
   - Not pursued: the disk cache is the library-first answer.
4. **Disk cache size:**
   - `f2` 2.9–4.6 MB per model;
   - `main` 6.3–15 MB, because `main` writes a new set of entries for every dataset.

## Closed and still UNVERIFIED
- **Closed:**
  - GPU timing of the vmapped ε0: warm 0.00–0.02 s;
  - memory at C=1,024, d=175: radon peaks at 1,172 MiB;
  - the GPU persistent cache: all programs hit on all 4 models, with new data.
- **UNVERIFIED:**
  - float32 end to end;
  - GPU timing of `jittered_initial_points`: `jitter+adapt_diag` is not the default, and no F2f row
    uses it.

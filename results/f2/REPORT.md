# F2 — final report (2026-10-24)

F2 makes `sample_pf_chees_hmc` compile once per model structure, and turns data and prior values into runtime
inputs. It ran from 2026-10-22 to 2026-10-24 in 11 work packages (F2a–F2k): fast-tier tasks, one at a time, reviewed
and merged by the lead. Each package has a SUMMARY in `results/f2/<WP>/`, and the running record is
`results/f2/LEDGER.md`.

**State:**
- branch `f2` = 25986e5;
- local `main` = 0d7157f, whose tree equals `f2`'s;
- not pushed: `origin/main` is still ecd9ba1;
- no work branches or checkouts are left;
- **no sampler default was changed after F2g**, and float32 is reported only.

## What a user gets
| | before F2 (`main` 232ae58) | after F2 |
|---|---|---|
| GPU warm call, 1,024 chains × 10 draws (8schools / gev / logit / radon) | 6.0 / 10.2 / 8.2 / 20.9 s | **0.21 / 0.47 / 1.39 / 6.4 s** |
| a call with new data or new prior values, same process | recompiles everything | **0 XLA and 0 PyTensor compiles** |
| first call in a fresh process, disk cache warm (GPU, new data) | not available | 3.6 / 6.2 / 4.9 / 9.6 s |
| GPU vs 32-core CPU NUTS, same accuracy (run 2's R2-T4) | 0.51–1.10× | **2.5–15×** (F2f) |
| R1 bias gate (b²_max < 0.01, seeds 1–3, all 4 models) | pass | pass, with identical draws on 3 models |

The warm calls are from this session's float64 rows (`F2k/gpu.md`), and they match F2f's. The fresh-process row is
F2g's. Calls 2 and later at the same shapes compile nothing, with the same or new data or priors.

**The rules are stated in graph terms, never per model.** Rule C lifts distribution parameters, observed data,
arrays and shared variables to runtime inputs before PyTensor's JAX rewrite. Formula constants stay static, so
PyMC's stability rewrites survive.

## The work packages
| WP | what | merged |
|---|---|---|
| F2a | the log-density takes model inputs as runtime arguments (rule C); `erfcx` registered to JAX's own, with a guard band | f2 |
| F2b-1, F2b-2 | the sampler builds once, with per-run values as arguments; the warmup length becomes a runtime trip count, and the draws need one program per `min(D, 1000)` | f2 |
| F2c, F2d | start-candidate scoring capped as d grows; the initial point without per-call PyTensor compiles | f2 |
| F2e | lift memo (structural hash, verified by `equal_computations`); opt-in disk cache | f2 |
| F2f | measurement on the GPU against `main`; R2-T4 redone; R1 gate 4/4 | report |
| F2g | jitted start-repair check (GEV GPU cold init 9.0 → 2.5 s); disk cache on by default; test teardown | main 780c3dc |
| F2h | `1 < pm_chains < chains` fixed (cyclic assignment; pre-existing bug) | main beb3087 |
| F2i | cheaper widening: the same graph, 0.016–0.046 s less per call | main 67f2820 |
| F2j | tests release compiled programs only above `max_map_count // 2 − 16,000` mappings: suite 774 → 725 s | main 89721fd |
| F2k | float32 runs end to end; measured on CPU and GPU | main 0d7157f |

Suite on the final head: `596 passed, 63 warnings in 752.89s (0:12:32)`. It runs on the CPU in one process.

## float32 (F2k): faster where it works, not safe to recommend yet
`PYTENSOR_FLAGS=floatX=float32` is PyMC's own switch. It had never run here, because the example notebooks turn
`jax_enable_x64` back on after PyTensor turns it off. The tools now switch it off again, and the sampler warns
about the contradiction.

On the A2000, at the gate shapes:

| model (W) | warm call, f64 → f32 | R1 gate under float32 |
|---|---|---|
| logit_1000x25 (50) | 1.39 → **0.18 s (7.7×)** | 3/3, ε and L as float64 |
| radon_noncentered (300) | 6.42 → **2.06 s (3.1×)** | 3/3, ε and L as float64 |
| 8schools (25) | 0.21 → 0.16 s | **fails 2 of 13 seeds** (float64: 0 of 13), both with one signature |
| gev (25) | 0.47 → 0.75 s | **the step size collapses** (3e-5–9e-3, R-hat 57–112). The chains barely move. |

On logit and radon, float32 would put the GPU at about 54× and 9× CPU NUTS, against F2f's 6.8× and 2.5–3.8×. GEV's
float32 gradient of PyMC's `GenExtreme` density is up to 18× off near ξ ≈ 0. That is measured, but is UNVERIFIED as
the whole cause.

## Decisions for the owner (nothing done)
1. **Push local `main` (0d7157f) to `origin`?** It holds all of F2 plus run 2.
2. **float32 in the README?** The recommendation is a short "float32 on the GPU" section:
   - it is opt-in;
   - it gives these speed-ups;
   - it states the GEV and 8schools caveats;
   - it tells the user to check ε and R-hat.

   Do not make it a default. A float32 default would also need a decision about the notebooks' x64 cell.
3. **Upstream reports, drafted and not filed:**
   - PyMC 5.28.5 `pm.Uniform(-100, 100)` gives logp NaN through int8 overflow (fixed on our path).
   - JAX `erfcx` returns 0 on [26.54, 26.64] at float64. The draft is in `F2a/SUMMARY.md`. F2k found the float32
     counterpart, [9.19455, 9.41928], which the guard already covers. That fact belongs in the same report.
   - Possibly PyMC `GenExtreme`'s float32 precision near ξ = 0, once its role in the collapse is verified.
4. **Possible follow-ups, none started:**
   - 8schools float32 at a longer warmup (does the failing signature go away?);
   - GEV float32: a graph-level stable form for `log1p(ξ z) / ξ` near 0, if the gradient error is the cause;
   - the pre-existing multi-block stop rule keeps only the last block's `WarmupInfo`, so gradients are
     under-counted.

## Process notes
- The owner's up-front answers covered the merges into local `main` and the follow-ups F2h–F2k. They are recorded
  in the ledger. No question was asked during the run.
- Two tasks (F2g and F2k) stopped at the time limit and were resumed, not restarted.
- TASK-RULES made fast-tier tasks sign commits with a pinned tier name up to F2i. The rule now records which pass produced the commit
  (62c38a3). Merged history is not rewritten.

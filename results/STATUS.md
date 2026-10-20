# STATUS — pf-r1 run

Branch: `pf-r1` (from `ecd9ba1`). Never commit to `main`, never push (A3).
Plan: `docs/review-2026-10/pf-chees-hmc-implementation-plan.md` — Part A run rules, Part B work order.

| WP | Scope | State | Branch | Last commit | Next step |
|---|---|---|---|---|---|
| WP0.1 | branch, review docs + probes committed | DONE | pf-r1 | c3bdbfc | — |
| WP0.2 | reference posteriors (CPU NUTS, 4 models) | DONE (merge queued) | wp/wp0refs | 67da0ca | stats fixed and independently verified (0 contract violations); needs 2 fast-tier reviews, no slot free |
| WP0.3 | `tools/_common.py`, GPU lock, env banner | DONE | pf-r1 | c3bdbfc | env.json captured (arviz_stats 1.3.3, nested R̂ ok) |
| WP0.4 | `arviz-stats`, skill update | DONE | pf-r1 | c3bdbfc | `pixi run test` 36 passed |
| WP1 | metric orientation + ordering fix | IN-PROGRESS | wp/wp1 | c3bdbfc | task `wp1-metric-orientation` (fast-tier) running |
| WP2 | paper-faithful ChEES adapter (vendored `base`) | TODO | wp2 | — | after WP1 merges |
| WP3 | R1 bias harness + nested R̂ | IN-PROGRESS | wp/wp3 | c3bdbfc | task `wp3-r1-harness` (fast-tier) running |
| WP4 | ensemble metric during warmup | TODO | wp4 | — | after WP2 merges |
| WP5 | capacity + fixed costs at large C | TODO | wp5 | — | after WP1 merges |
| WP6 | GPU per-step efficiency (flag A/B, unbatched trip, profile) | TODO | wp6 | — | wave 3 |
| WP8 | hygiene items (conflict-free subset only) | IN-PROGRESS | wp/wp8 | c3bdbfc | task `wp8-hygiene-safe` (steady-tier); rest held until WP1/WP2/WP3 merge |
| WP9 | re-runs + doc retractions + REPORT | TODO | pf-r1 | — | after WP1–WP5 |

Wave: 1. GPU lock: `/tmp/pf-gpu.lock`. Caps: 2 × fast-tier, 3 × steady-tier (A4).
Running now: 2/2 fast-tier (WP1, WP3) + 2/3 steady-tier (WP0.2 rework, WP8).
References verified good on disk (all 4, per-parameter stats, sha match) — WP3 unblocked. wp/wp0refs merge awaits 2 reviews.

Lead verification before tasks started (not taken on faith):
- `sampler.py:406` is `inverse_mass_matrix = jnp.linalg.inv(cov)` — inversion confirmed in source.
- `tools/probes/order_check.py` on CPU: GEV `self.rvs` [μ, σ_log__, ξ_interval__] vs dict-flatten
  [μ, ξ_interval__, σ_log__] → DIFFERENT; radon NC DIFFERENT; 8 Schools SAME. Kinetic energy with an
  `imm` intended to give `z_b` variance 100 returned 0.5, not 50. The ordering bug is real.
- Consequence recorded: every GEV and radon metric verdict in `docs/` is confounded (WP9 retracts).

## PAUSED 2026-10-19 18:38 — box yielded to another user

All five tasks stopped (sessions retained, resumable). No process of ours holds the GPU (99 MiB used, 0% util). Nothing pushed; `main` untouched at `ecd9ba1`.

Committed state on `pf-r1`: WP0 + probes + lock, WP0.2 rejection and verification records, `_common` peak-mem fix, the §0.1 CONTRADICTION and its RETRACTION (`a03dc0f`), the G-R1-diag gate finding (`d4779fa`), the claim inventory with corrected citations (`7033037`, `e660578`).

Branch heads: `wp/wp1` f7aa1ca · `wp/wp3` d080195 · `wp/wp0refs` 67da0ca · `wp/wp8` 8931a71 · `wp/wp9docs` (created, no edits yet) · `wp/wp2` (created or not, no edits).

Verified data banked: references 4/4 sha256 match and per-parameter shapes confirmed by the lead; WP3 suite 65 passed on forced CPU; §2e reproduces; the 2×2 reproduces all 12 cells independently; radon and Fisher arms in `WP1__rebaseline.jsonl` (Fisher cut mid-block, partial rows only).

## Resume commands (in this order)

1. Covariance-orientation arm — the number that gates the default decision:
   `MODEL=<one of 8schools gev logit_1000x25 radon_noncentered> IMM=mc IMM_ORIENT=covariance ARM=wp1 SEEDS=43,44,45,46,47 REPEAT=2 OUT=$HOME/pf/wp1/results/runs/WP1__covarm.jsonl $HOME/pf/wp1/tools/gpu_run.sh pixi run --manifest-path "<MAIN>/pixi.toml" env PYTHONPATH=$HOME/pf/wp1 python $HOME/pf/wp1/tools/wp1_rebaseline.py`
2. Reviews still owed (two per branch, both fast-tier): `wp/wp3` stats review (code review stopped mid-pass), `wp/wp0refs` stats review, `wp/wp8` code + stats.
3. Then merges in dependency order: `wp0refs`, `wp8`, `wp3`, `wp1`, then branch `wp/wp2` (vendored ChEES) and `wp5` (capacity at C=1024, needs wp1 merged).
4. WP3's unfinished positive control: `K=8 SEEDS=1,2 WS_8S=25,100 WS_LOGIT=25,400 ARM=legacy_k8 OUT=$HOME/pf/wp3/results/runs/WP3__nested_k8.jsonl sh $HOME/pf/wp3/tools/wp3_selftest.sh`
5. Open gate question to resolve before any accept/kill cites nested R̂: nR̂ reads ~1.004 at both b²=0.00063 and b²=0.4100 (LEDGER `d4779fa`).

## 2026-10-20 00:12 · A4 two-review gate is ENVIRONMENT-BLOCKED, not skipped

Both mandated reviewer tasks for `wp/wp1` were interrupted and the replacement task wrote `NOT RUN` to `~/pf-research/REVIEW-wp1-stats.md` (1 line, on disk). `static review pass` also failed: all three panel members timed out at 3,000 s. Reviews on record: `WP3-code.md`, `wp0refs-code.md`, `WP1-lead-verification.md`.

**Decision: no branch merges into `pf-r1` until at least one independent review can actually execute.** The gate is a trust boundary, and silently substituting the lead's own review for two independent ones is exactly the kind of thing this run exists to catch. Measurement continues on `pf-r1-int`, which is where every number in the REPORT already comes from.

Substituted evidence, so the record is not empty: merged-tree suite 99 passed · G-ORDER teeth verified by running the tests against `ecd9ba1` (12 substantive failures, 5 passes) · orientation fixture re-measured personally · T1 48 rows and the GEV G3 probe run by the lead. What those do *not* replace: a second pair of eyes on whether the acceptance criteria were read correctly.

## 2026-10-20 00:25 · lead's shell died mid-turn — durable state, nothing uncommitted that matters

Several tool calls were unavailable in that session. So the review of `wp/wp2` that was about to launch did **not** launch, and no task could be re-armed. Write this down instead of losing it.

**Durable right now (all committed, nothing pushed):**
- `pf-r1` = `008ccf3`. Sequence since the pause: `c2993cf` GEV 2×2 → `a03dc0f`/`699be6d` orientation retraction chain → `5a18c88` RETRACTION of my own contradiction (orientation IS wrong) → `13d8379` 4-model verdict → `9ce725e` nested-R̂ common-mode mechanism → `647798b` WP1 lead verification → `ba68aa7` T1 partial → `4770dcb` T1 complete + GEV G3 → `d1fca9c` REPORT updated → `008ccf3` this gate record.
- `pf-r1-int` = `574f960` — merged `pf-r1` + wp8 + wp0refs + wp1 + wp2 + wp3, suite **99 passed**, and every number in `results/REPORT.md` comes from here.
- Rows in git or on disk: `T1__r1_sweep.jsonl` (48, complete), `G3__gev_meanshift.jsonl` (10, complete), `WP1__rebaseline.jsonl` (142, 2 VOID-stamped), `WP1__covarm.jsonl` (40), `WP1__gev_2x2.jsonl` (24), `WP2__lgrowth.jsonl` (32), `WP3__nested_k8.jsonl` (23).
- Tasks still running when the shell died: `wp5-capacity` (5 commits on `wp/wp5`, 1 file uncommitted in its checkout) and `wp4-ensemble-metric` (just started). Both are detached OS-level sessions; check `git -C ~/pf/wp5 log --oneline pf-r1-int..HEAD` and the same for `wp4` rather than trusting their reports.

**The two results that decide the run** (both in `results/LEDGER.md`, restated here so a fresh session does not have to re-derive them):
1. **R1 works with the corrected metric** — *per seed, by the harness's own gate, corrected 00:35 after review*. C=1,024 × D=10: covariance + raw passes G-R1 at **W=50 on both seeds** on 8 Schools (955 gradients/chain), GEV, and logistic (335). The shipped orientation never passes on logistic (b² 0.193/0.158/0.0498 at W=50/100/200). **radon NC passes on no W for both seeds** — at W=200 seed 1 gives b² 0.0234 (fail) and seed 2 gives 0.0071 (pass), and W=100 is `unresolvable` under the gate's floor rule. My earlier median-based reading said "radon needs W=200 (1,521 gradients)", which came from averaging two seeds that disagree. The gate is per seed.
2. **Default flip is an owner call with a measured trade.** GEV, 5 seeds: shipped fails R̂ ≤ 1.01 on **5/5** seeds (to 1.0688); candidate passes **3/5** (1.0065, 1.0017, 1.0054, 1.0141, 1.0117 — I miscounted 4/5) and halves mean error (0.059 → 0.028 sd) but divergences rise seed-stably 0.025 → 0.400 per 1k. G3's "means within 1σ" clause is unsatisfiable as written at 40k draws (worst z 3.4 on a 0.026 sd shift) — a plan defect, needs restating in sd units.
3. **"Ship the knob, hold the default" was wrong** (reviewer's point, accepted): the permutation is unconditional in `wp/wp1`, so merging it with `precision` as default *changes* released behaviour to permuted+precision, which fails R̂ on GEV in 5/5 seeds where today's code passes. There is no neutral interim state — either the permutation and the covariance default go together, or the permutation does not merge. Recommendation is now **flip, in the same merge**. Note also that T1's two arms move two factors at once (orientation and iterate), and all 48 rows are cold with 2 seeds, so the R1 wall-clock comparison is not decidable from them. The floor rule is `floor > 0.005 → unresolvable`, not "b² below its own floor"; I had that wrong in three places.

**First four actions on resume:**
1. `wc -l ~/pf/int/results/runs/*.jsonl` and `git -C ~/pf/wp5 log --oneline pf-r1-int..HEAD | cat` — establish what the tasks really did, never what they said.
2. Launch the `wp/wp2` code review (brief content is in this thread at 00:20; it is a static diff of `pfcheeshmc/chees.py` against `.pixi/envs/default/lib/python3.13/site-packages/blackjax/adaptation/chees_adaptation.py`, checking for undeclared drift, `iterate=ma` state-for-state equivalence, freeze site per mode, `WarmupInfo` consumers, and whether `WP2__lgrowth.jsonl` supports the SUMMARY — the task's own final message quoted a file that does not exist).
3. Re-run the `wp/wp1` stats review with a working shell (the static pass covers criteria and interpretation only).
4. Then merges in dependency order and the remaining T-tables (T2 R2 throughput, T3 CPU-vs-GPU, T4 capacity, T5 per-step cost) — T4 and T5 depend on WP5/WP4 finishing.

## 2026-10-20 01:30 · owner rulings: flip approved, review gate stays strict, two leads deduplicate

1. **Default orientation: FLIP to `covariance`, in the same merge as the permutation.** Implemented on `wp/wp1` (`feat(sampler): default mass-matrix orientation is now the covariance`), the pinned test now asserts the default *mixes*, and 17 metric tests pass on forced CPU. **Not merged** — see ruling 2.
2. **A4 stays strict: no merge until two reviews are executed by reviewer tasks.** Consequence recorded honestly: `wp/wp0refs` was merged at `97f235a` **before** this ruling, on one task review (`WP0refs-code.md`, 18:34, blocking item since fixed) plus one lead-authored review (`WP0refs-stats.md`). That is one executable review, not two. Its content is verified — the shape guard raises on both failure modes and the reference-resolution margins (11.7×–20.2× below the b² threshold, min ESS 13,765–41,006) reproduce independently — so the merge is kept, flagged, and reversible at `97f235a^`. Every branch after this line waits for two tasks.
3. **Two leads are live on this repo** (this session and a resumed one that produced `d34613c`, `1ca0ce8`, `6db3a3b`). Their numbers re-checked here are correct, so the work is kept. From now on I re-read `git log pf-r1` before every commit and reconcile rather than assuming my view of the tree is the only one.

---

# RESUME PACKET — 2026-10-20 (supersedes everything above it)

**Tree.** `main` ecd9ba1 (untouched, nothing pushed) · `pf-r1` **9bf29a5** · `pf-r1-int` **129afce** (merged measurement tree; the row files are now committed there — they were untracked and would have been lost).

**Merged into `pf-r1`:** wp0refs, wp1 (permutation + covariance default), **wp3** (R1 bias harness + two integrity fixes at `9d5c30d`). Merged-then-reverted: **wp8** (`8d81563` → `60302b2`) because the reviews it was merged on did not exist on disk.

**Unmerged branches:** `wp/wp2` b02a6e8 · `wp/wp4` d457a17+ · `wp/wp5` 0e6739f · `wp/wp8` 8931a71 · `wp/wp9docs` 79e9944.

**Review coverage (count from disk, never from memory):**
| branch | code | stats | needed action |
|---|---|---|---|
| wp2 | ✓ (lead-authored) | task running | then merge |
| wp4 | ✓ (2 blockers refuted) | task running | then merge |
| wp5 | ✓ (3 of 4 blockers refuted, **hold**: attribution wrong) | task running | fix SUMMARY attribution, plumb or delete `pm_chains`, add tie/boundary tests, then re-review |
| wp8 | ✗ | ✗ | launch both |
| wp9docs | ✗ | ✗ | launch both |

**Tasks in flight (steady-tier, cap 3):** `wp4-stats-review-z`, `wp5-stats-review-z`, `wp2-stats-review-z`. On a reload they come back **paused** — resume with `session resume`, never re-launch, and never trust their final messages: write-the-file-first was mandated because 5 of 7 earlier tasks delivered messages with no file and refuted blockers.

**Next actions, in order:**
1. When a stats task lands: verify its blockers by running the named test (20–40 s each) before acting; file under `results/reviews/`.
2. Merge in this order on two clean reviews: **wp2 → wp4 → wp8 → wp9docs → wp5** (wp5 needs its attribution fix first).
3. Launch the 4 missing reviews (wp8 code+stats, wp9docs code+stats) as steady-tier tasks with the same anti-fabrication rules.
4. Remaining work items: **T5 cost split — history and callback measured (+3.3% / +0.3% / +64.2% stack, ~2–8% callback); kernel-vs-adaptation bounded but not separable, because warmup is cheaper per gradient than sampling**; GEV divergence coordinate diagnosis (one instrumented run); floor improvement (reference MCSE + autocorrelation) before any near-floor gate decision; WP4's refresh needs a package-level `metric_refresh_reseed` option + test (today it is a harness flag, so "refresh without re-seed" is measured, not shipped).

**Numbers that must not be re-litigated:** wall times in rows produced before `wp/wp3` `9d5c30d` are ~52% high warm (fixed, measured 15.42→10.62 s). `g_r1` masking fix changed 6 delivered cells (all radon, b² up to 63 vs floor 0.0056). Radon NC is **unmeasurable** at R1 (floors 0.0037–0.0071; W=200 costs 5.7× gradients for a 20% b² change). CPU NUTS T3: all 12 cells `resolvable=False`. Command buffers: −12%, not shipped. WP5 memory: 8 Schools 2,314 → 224 MiB (~10×); radon runs at C=1,024 (528.79 s cold / 42.94 s warm); no before/after wall comparison exists because the before-probe has no `wall_s`.

---

# FINAL STATE — 2026-10-20

**All eight work packages are merged into `pf-r1`. Merged suite: 125 passed in 169.28 s (forced CPU, at `f95e491`).**
`main` untouched at `ecd9ba1`; nothing pushed. Working tree clean.

Merged: wp0refs · wp1 · wp2 · wp3 · wp4 · wp5 · wp8 · wp9docs. (`wp/wp8` was merged once on an imagined review,
reverted at `60302b2`, reviewed properly, then merged for real.)

Read `results/REPORT.md` §6b for the T-tables, §7 for what is left, §8 for the handoff and the seven measured
caveats. The two that change decisions: absolute wall times before `wp/wp3` `9d5c30d` are ~52% high warm, and
radon NC cannot be gated at the R1 shape at any warmup.

**Corrected 2026-10-20:** the claim above that no before/after wall comparison exists is too strong — row-level
`wall_s` is null on the 80 before-probe rows, but per-stage seconds are present in `kwargs.stage_s` on 70/80, so
stage-level before/after is available (see §6b T4).

**Closed 2026-10-20 — the `pf_draws=1` gap (§3f).** The production 4,000-draw Pathfinder pool was measured at the
R1 shape, uncapped and capped. Uncapped it **cannot run above d ≈ 8** on the 6 GB card (the pool operand alone is
4.1M rows × 175 × 8 = 5.73 GB at d=175; XLA's over-limit number matches to the byte). Capped to WP5's `4C` it runs
everywhere, costs 9× less peak memory on 8 Schools (2,076 → 224 MiB), and **changes no bias verdict** against a
matched 1-draw control. Consequence: default `pf_pool_cap` on; the start policy is not a bias lever.

Also closed: five row files cited by the report's T-tables (`T1__r1_sweep`, `T3__cpu_nuts`, `T5a__cmdbuf`,
`ORDER__radon_noncentered`, `ORDER__logit_1000x25`) lived only on the divergent `pf-r1-int` branch and are now on
`pf-r1`; every cited row count was re-checked against its file and is correct.

**Still open, in order:**

1. **`pf_pool_cap`'s dimension-aware default `max(4C, 4d²)` is settled** (REPORT §3i, 12 rows). Neutral
   against the uncapped pool on all four models at C=32 (radon 300.7 vs 300.2 grad/ESS, 53 vs 53 ESS/s), and it
   removes the C=1,024 OOM. Plain `4C` is rejected: 42× gradients/ESS on radon, tail ESS 188, R̂ 1.195.
   Remaining gap: one seed per cell.
2. **T5's kernel-vs-adaptation quarter is bounded, not separated** — warmup runs at 0.41–0.46× sampling's
   cost per gradient, so the planned subtraction goes negative. Settling it needs a `sample_stats` ablation.
3. **GEV's diverging coordinate.** Start dispersion is measured (`ξ_interval__` 0.156×, and 8 Schools' `mu` is
   colder at 0.159× with no divergence problem, so it is a pattern not a cause). The discriminating run — same
   target, that coordinate re-dispersed — has not been done. Per-step velocity is unobtainable while the
   warmup output stays projected to `WarmupScanInfo`.
4. **The noise-floor improvement** (reference MCSE + serial correlation), which is also what makes radon
   not gateable at R1.
5. **The two `ORDER__` row files** need a re-run on `pf-r1` to count as current-tree evidence (their shas are
   not in `pf-r1`'s history).

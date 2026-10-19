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

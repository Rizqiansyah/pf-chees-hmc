# STATUS — pf-r1 run

Branch: `pf-r1` (from `ecd9ba1`). Never commit to `main`, never push (A3).
Plan: `docs/review-2026-10/pf-chees-hmc-implementation-plan.md` — Part A run rules, Part B work order.

| WP | Scope | State | Branch | Last commit | Next step |
|---|---|---|---|---|---|
| WP0.1 | branch, review docs + probes committed | DONE | pf-r1 | c3bdbfc | — |
| WP0.2 | reference posteriors (CPU NUTS, 4 models) | IN-PROGRESS | wp/wp0refs | 82bb771 | **rejected on re-verification**: vector-variable stats collapsed over the parameter axis; task re-running with a shape-guard test |
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
References are on disk (all 4, gate PASS on R̂/ESS) but the branch is not merged until the stats fix lands.

Lead verification before tasks started (not taken on faith):
- `sampler.py:406` is `inverse_mass_matrix = jnp.linalg.inv(cov)` — inversion confirmed in source.
- `tools/probes/order_check.py` on CPU: GEV `self.rvs` [μ, σ_log__, ξ_interval__] vs dict-flatten
  [μ, ξ_interval__, σ_log__] → DIFFERENT; radon NC DIFFERENT; 8 Schools SAME. Kinetic energy with an
  `imm` intended to give `z_b` variance 100 returned 0.5, not 50. The ordering bug is real.
- Consequence recorded: every GEV and radon metric verdict in `docs/` is confounded (WP9 retracts).

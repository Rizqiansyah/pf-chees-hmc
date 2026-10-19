# STATUS — pf-r1 run

Branch: `pf-r1` (from `ecd9ba1`). Never commit to `main`, never push (A3).
Plan: `docs/review-2026-10/pf-chees-hmc-implementation-plan.md` — Part A run rules, Part B work order.

| WP | Scope | State | Branch | Last commit | Next step |
|---|---|---|---|---|---|
| WP0.1 | branch, review docs + probes committed | IN-PROGRESS | pf-r1 | — | commit |
| WP0.2 | reference posteriors (CPU NUTS, 4 models) | TODO | wp0refs | — | launch steady-tier after WP0 commit |
| WP0.3 | `tools/_common.py`, GPU lock, env banner | IN-PROGRESS | pf-r1 | — | generate env.json |
| WP0.4 | `arviz-stats`, skill update | TODO | pf-r1 | — | pixi add + `pixi run test` |
| WP1 | metric orientation + ordering fix | TODO | wp1 | — | launch fast-tier after WP0 |
| WP2 | paper-faithful ChEES adapter (vendored `base`) | TODO | wp2 | — | after WP1 merges |
| WP3 | R1 bias harness + nested R̂ | TODO | wp3 | — | launch fast-tier after WP0 |
| WP4 | ensemble metric during warmup | TODO | wp4 | — | after WP2 merges |
| WP5 | capacity + fixed costs at large C | TODO | wp5 | — | after WP1 merges |
| WP6 | GPU per-step efficiency (flag A/B, unbatched trip, profile) | TODO | wp6 | — | wave 3 |
| WP8 | hygiene items | TODO | wp8 | — | any free steady-tier slot |
| WP9 | re-runs + doc retractions + REPORT | TODO | pf-r1 | — | after WP1–WP5 |

Wave: 1. GPU lock: `/tmp/pf-gpu.lock`. Caps: 2 × fast-tier, 3 × steady-tier (A4).

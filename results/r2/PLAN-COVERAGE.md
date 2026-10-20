# PLAN-COVERAGE — run 2

One row per numbered item in `docs/review-2026-10/pf-chees-hmc-run2-plan.md` Part B, plus the A3/A7/A9
contract items. An item that is silently absent counts as a run failure (plan A4). Statuses:
`DONE (sha/rows)`, `SKIPPED (reason)`, `DEVIATED (what and why)`, `PENDING`, `CONDITIONAL`.

60 items tracked. Updated as the run proceeds.

| id | item | owner | status |
|---|---|---|---|
| A3 | A3 branch/env/hygiene first commits | lead | DONE 934e576 |
| LEAD | Lead workflow (owner, 2026-10-20) | lead | DEVIATED (owner request; LEDGER handover entry) |
| WP3-split | R2-WP3 run as 3a (new files) + 3b (r1_bias.py) tasks | lead | DEVIATED (wall time; files disjoint) |
| A7 | results/r2/ skeleton + env.json | lead | DONE |
| WP0.1 | R2-WP0 item 1: A3 commands | lead | DONE 934e576 |
| WP0.2 | R2-WP0 item 2: results/r2 skeleton + env.json | lead | DONE |
| WP0.3 | R2-WP0 item 3: baseline suite on pf-r2 before merges | lead | DONE 4e07b45: "125 passed in 165.79s (0:02:45)" |
| WP0.4 | R2-WP0 item 4: quarantine tools/legacy/t3_cpu.py | lead | DONE |
| WP1.1 | R2-WP1 change VERBATIM: whitened lambda_max re-seed + clip + new signature + attrs | r2/wp1 | DONE 9742363 (merged 07575bc) |
| WP1.2 | R2-WP1 test 1: unit invariance (rescale coord 0 by 10, coord 3 by 0.01 -> T0 unchanged 1e-9) | r2/wp1 | DONE 9742363 |
| WP1.3 | R2-WP1 test 2: identity B=S -> T0=pi/2 (1e-8) | r2/wp1 | DONE 9742363 |
| WP1.4 | R2-WP1 test 3: d=2, B=diag(S) -> T0=(pi/2)sqrt(1+rho) (1e-8) | r2/wp1 | DONE 9742363 |
| WP1.5 | R2-WP1 test 4: reject non-PD B (keep existing) | r2/wp1 | DONE 9742363 |
| WP1.6 | R2-WP1 test 5: delete test_reseed_matches_closed_form (pins the defect) | r2/wp1 | DONE 9742363 |
| WP1.7 | R2-WP1 CPU smoke: 8schools+logit C=64 D=10 W=50, reseed on/off, L within 0.5-2x | r2/wp1 | DEVIATED 9942ac0: band missed on logit (3.07x), not A6 BLOCK; LEDGER |
| WP1.8 | R2-WP1 mutation proof: test 1 fails against the OLD formula | lead | DONE (task mutation: 3 failed; SUMMARY) |
| WP2.1 | R2-WP2 item 1: imm_source="pf_closed" via lbfgs_inverse_hessian_formula_1, no ridge, PD assert | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.2 | R2-WP2 item 2: pool sizing conditional on imm_source (non-mc -> 4C) | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.3 | R2-WP2 item 3: do NOT change imm_source default (D2 decides after M1) | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.4 | R2-WP2 item 4: pf_start_scale kwarg, 1.0 bit-identical | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.5 | R2-WP2 item 5: idata.attrs["config"] write (resolved sampler config) | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.6 | R2-WP2 test 1: exact algebra vs lbfgs_inverse_hessian_to_low_rank_metric, 4 models | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.7 | R2-WP2 test 2: MC consistency N=50,000 on 8schools+gev | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.8 | R2-WP2 test 3: G-ORDER fixture passes with pf_closed; fails when choke point mutated | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.9 | R2-WP2 test 4: pool rows 4,096 at C=1024 for pf_closed vs max(4C,4d^2) for mc | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP2.10 | R2-WP2 test 5: pf_start_scale 1.0 bit-identical, 2.0 doubles start sd (1e-12) | r2/wp2 | DONE (merged 62c7620; SUMMARY) |
| WP3.1 | R2-WP3 item 1: defaults pass through (None -> sampler decides) + effective recorded + disagree -> raise | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP3.2 | R2-WP3 item 2: --pool-seeds pooled row (G-R1-pool diagnostic, never a gate) | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP3.3 | R2-WP3 item 3: --metric-refresh-warmup recorded | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP3.4 | R2-WP3 item 4: tools/t3_cpu_nuts.py (score_ensemble, include_transformed, exact grads incl. tune) | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP3.5 | R2-WP3 item 5: summarise_rows.py smallest-pass | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP3.6 | R2-WP3 item 6: document grads_per_chain_total composition + pf_grads_total | r2/wp3 | DEVIATED 353c02f: documented as is; total excludes eps probes + fisher pool grads (SUMMARY 3b) |
| WP3.7 | R2-WP3 test 1: floor regression +/-30% of 2ln(10)/(C.D), squared units, via both scorers | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP3.8 | R2-WP3 test 2: passthrough -> effective.imm_orientation == covariance with no flags | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP3.9 | R2-WP3 test 3: pooled seeds same b2, floor/3 (+/-5%) | r2/wp3 | DONE (merged 353c02f; R2-WP3a/3b SUMMARY) |
| WP4.1 | R2-WP4 item 1: apply E1-E7 inline | r2/wp4 | DEVIATED 25d2618: E6 deleted (no row file), E3c narrowed (plan wrong) |
| WP4.2 | R2-WP4 item 2: Errata (run 2) block E1-E12 + inline void tags | r2/wp4 | DONE (merged 25d2618; R2-WP4 SUMMARY) |
| WP4.3 | R2-WP4 item 3: do not rewrite run-1 numbers | r2/wp4 | DONE (merged 25d2618; R2-WP4 SUMMARY) |
| WP4.4 | R2-WP4 item 4: results/r2/RUN1-COVERAGE.md | r2/wp4 | DONE (merged 25d2618; R2-WP4 SUMMARY) |
| WP4.5 | R2-WP4 item 5: confirm .gitignore (, ref clean) | lead | DONE |
| M1 | M1 closed form vs MC covariance (R2 3 seeds, R1 +pool-seeds, radon memory) | steady-tier runner | DONE (eee10ae trees; M1__{r2,r1,mem}.jsonl, 18 jobs rc=0) |
| D2 | D2 rule evaluation (a)-(d) recorded row by row | lead | DONE: NO (clause b2 fails on radon 2 vs 3; LEDGER) |
| M2a | M2a R1 iterate sweep: 4 models x 2 arms x W{25..300} x 3 seeds = 120 cells | steady-tier runner | RUNNING |
| M2b | M2b R1 refresh with fixed re-seed (diag / diag_norese / fisher) | steady-tier runner | PENDING |
| D3 | D3 recommendation only (default stays off) | lead | PENDING |
| M2c | M2c start scaling, conditional on radon having no passing W<=300 | steady-tier runner | CONDITIONAL |
| M3 | M3 R2 matrix 3 arms x 5 seeds = 60 cells + mean shift in sd units + G3 on GEV | steady-tier runner | PENDING |
| D1 | D1 rule evaluation (a)-(c) | lead | PENDING |
| M4 | M4 GPU warm timing at R1 operating point (repeat 2, rep-1 rows; + C=2048 scaling) | steady-tier runner | PENDING |
| M5 | M5-cpu CPU NUTS time-to-pass: 4 models x 3 tune x 4 draws x 3 seeds = 144 runs | steady-tier runner | PENDING |
| T1 | R2-T1 R1 smallest passing W | lead | PENDING |
| T2 | R2-T2 M1 closed vs MC + D2 | lead | PENDING |
| T3 | R2-T3 M3 R2 matrix + GEV G3 + D1 | lead | PENDING |
| T4 | R2-T4 time to pass CPU vs GPU (headline) | lead | PENDING |
| T5 | R2-T5 decisions D1/D2/D3 with clause-by-clause evaluation and deciding row ids | lead | PENDING |
| T6 | R2-T6 errata applied + coverage counts | lead | PENDING |
| OPEN | Open questions for the owner: at most 5, each with the settling measurement | lead | PENDING |
| A9 | A9 done-checks + print branch, log range, REPORT path | lead | PENDING |
| E13 | New run-1 defect: single-phase _run_chees ignored iterate (found in R2-WP2) | r2/wp2 | DONE 5982ee7; LEDGER |

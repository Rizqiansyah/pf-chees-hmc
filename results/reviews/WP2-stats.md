# WP2 — statistical-validity review (the raw vs moving-average iterate)

**Provenance: lead-executed, NOT independent.** The assigned task died with a broken tool layer, twice;
its earlier report cited four commit hashes (`58d582c`, `490d845`, `956d26f`, `066986c`) — `git cat-file -t`
returns *"Not a valid object name"* for **all four**, and it claimed "two blockers already fixed by the
lead", which never happened. So nothing from that task is used here. Every figure below was printed in
this session from `~/pf/wp2/results/runs/WP2__lgrowth.jsonl` (32 rows) by `~/scratch/wp2_verify.py`.

**One correction that matters before the numbers:** the claim that the vendored `chees.py` "accepts `optim=None`
and crashes" is only half true — `chees.py:188` reads `if optim is None: optim = optax.adam(adam_lr, ...)`, so
it **silently substitutes Adam** rather than raising. A silent substitution in a vendored numerical file is
arguably worse than a crash, so this stays as a finding even though the task's description of it was wrong.
`chain_group_fn` is not present in the vendored file at all (0 matches), so that part of its report is moot.

## 1. Frozen values, per seed, as delivered

| model | arm | `frozen_L_nominal` (seed 43, 44) | `frozen_eps` | wall warm | wall cold |
|---|---|---|---|---|---|
| logistic | `A_prec_ma` | 116.3, 130.2 | 0.0021, 0.0021 | 78.76, 16.72 | 78.34, 18.18 |
| logistic | `B_cov_ma` | 26.8, 27.5 | 0.6548, 0.6476 | 27.42, 7.30 | 27.56, 7.28 |
| logistic | `C_cov_raw` | 25.0, 26.0 | 0.6419, 0.6485 | 31.43, 7.86 | 31.37, 7.78 |
| logistic | `D_cov_raw_t0` | 24.9, 24.2 | 0.6565, 0.6488 | 31.96, 7.92 | 31.83, 7.88 |

(8 Schools is the same shape: A 44.53 @ ε 0.177, B 21.40 @ 0.524, C 21.67 @ 0.513, D 19.78 @ 0.543.)

## 2. The accept rule is **NOT ESTABLISHED**, not PASS

The plan kills `raw` if its frozen L is worse than `ma` at W=2,000 on **two or more** models. Measured:
logistic 25.0 vs 26.8 (raw better by 6.7%) and 26.0 vs 27.5 (better by 5.5%); 8 Schools 21.67 vs 21.40 (raw
**worse by 1.3%**) and 19.78 vs 21.5 (better by 8%). So one model shows a regression, and it is 1.3%.

But there are **two seeds per cell**, and the seed-to-seed spread *within* an arm is larger than the
arm-to-arm difference it is supposed to rank: logistic `A` spans 116.3–130.2 (12%), `B` 26.8–27.5 (2.6%). A
1.3% difference between two 2-seed medians is not separable from that. **Verdict: NOT ESTABLISHED** —
"ACCEPT as an option, default unchanged" is a defensible engineering call, but it is not what these rows prove.
Two more seeds on the 8 Schools pair would settle it.

## 3. The wall saving is real only when paired by seed

Every arm is bimodal on logistic: seed 43 takes ~78 s and seed 44 ~17 s in arm A, ~31 s and ~7.8 s in arm C.
Comparing medians across differently-ordered pairs would be meaningless. **Paired per seed**, A→C is
78.76 → 31.43 (**2.5×**) and 16.72 → 7.86 (**2.1×**), so the ~2.1× warmup saving is genuine on that model —
and it belongs to the *metric* arm change (A is precision, C is covariance), not to the iterate:

`L·ε` per arm, logistic: A 0.241 → B 17.55 (**73×**) → C 16.05 (**8.5%**) → D 16.35.
8 Schools: A 7.88 → B 11.22 (1.42×) → C 11.11 (1%) → D 10.74.

That is the branch's best argument and the rows carry it cleanly: **the metric sets the frozen operating point;
the iterate changes convergence speed.** Anyone budgeting the raw-iterate change against the frozen point is
mis-attributing the gain — and the plan's §2c framing invites exactly that. Note the caveat from §5: these
walls also predate the harness fix at `wp/wp3` `9d5c30d`, so absolute seconds are inflated by the per-step
diagnostic (measured ~52% warm); ratios at equal warmup cancel.

## 4. Reproduction against the plan's targets

Plan §2a/§2c versus these rows (seed 43, the task used seed 43; the plan used 42): logistic A 0.0021/136 vs
0.002083/116.3; B 0.66/26 vs 0.6548/26.8; C 0.66/27 vs 0.6419/25.0; 8 Schools A 0.18/43 vs 0.177/44.5,
B 0.51/22 vs 0.524/21.4, C 0.53/19 vs 0.513/21.7 (L +14%, within the seed spread above).
**Seven of eight frozen-ε pairs match to under 3%; the one L miss is inside seed noise.** The reproduction is
sound.

## 5. Test strength

`tests/test_chees_vendored.py` was re-run under the code review: **9 passed in 14.9 s** forced CPU, and the
equivalence test pins **64 steps of state identity** between `iterate="ma"` and the installed adapter — which
is the property that makes "vendor-and-fix" reviewable, not just "same final answer". One asymmetry worth a
test: in `ma` mode the trace's trajectory length is reported from the moving average, in `raw` mode from the
iterate, so the two arms' `L_at` curves are not the same quantity at the same step index. That is cosmetic at
the freeze and misleading in a plot.

## Verdict

**BLOCKING:** none.
**NON-BLOCKING:**
1. State the accept decision as NOT ESTABLISHED at n=2 seeds, and add two seeds on the 8 Schools pair.
2. `optim=None` silently substitutes Adam instead of raising — make it explicit or document the substitution.
3. Trace semantics differ between modes (MA vs iterate); one sentence in the docstring and one test.
4. Absolute walls in these rows are inflated by the pre-`9d5c30d` harness; quote the paired ratios.
**NIT:** the branch's `results/WP2/SUMMARY.md` reproduces the plan's numbers faithfully — the fabrications were
in the *review task's* report, not in the branch's.

> **RECOMMEND: merge** — default stays `ma`, `raw` ships as a measured option whose convergence-speed advantage
> is real (L@100 of 25–42 vs 8.3) and whose effect on the frozen point is inside noise.

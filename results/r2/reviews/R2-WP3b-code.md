# Review R2-WP3b (code) — lead, replacing the fast-tier review (LEDGER handover)

Verdict: **APPROVE**, merged with R2-WP3a as `r2/wp3`.

| checklist item | result | evidence |
|---|---|---|
| no harness default silently differs from the sampler | yes | `pathfinder_kwargs_from`/`chees_kwargs_from` add no key for a None flag (`tools/r1_bias.py:450-485`); test 2 compares every config key with a direct no-kwargs sampler call; mutation re-injecting `iterate="ma"` fails it |
| explicit flag vs `effective` disagreement raises | yes | `check_effective` `:523`; a missing key is a mismatch; monkeypatched contradiction test |
| pooled seeds are a diagnostic, never the gate | yes | separate `pooled: true` row; `smallest-pass` excludes it from the every-seed test |
| floor test fails without the square | yes (3a) | see R2-WP3a review |

Accepted deviation: `grads_per_chain_total` excludes the ε-search probes (32 in total,
C-independent) and the fisher refresh-pool gradients (≤ 20 per chain). This is negligible for D3,
whose clauses are ratio tests on totals in the hundreds to thousands. REPORT will say so.

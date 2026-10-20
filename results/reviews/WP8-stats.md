# WP8 — statistical-validity review (does the new test detect anything that actually went wrong?)

**Provenance: lead-executed, NOT independent** — flagged as an A4 deviation in `results/STATUS.md`.
Every number here was run in this session; the decisive result is item 3.

## 1. Is mean+variance within 4 MCSE the right criterion?

As a *pipeline sanity* gate, yes: mean and variance of a known analytic posterior are the two lowest-cost
moments that a broken transform, a mis-set Jacobian, or a collapsed coordinate will move, and 4 MCSE from an
autocorrelation-aware MCSE is roughly a 1-in-10,000 false-alarm rate per parameter under independence (looser
with correlated draws, which is why using `arviz_stats.mcse` rather than `sd/√n` matters). What it is **not** is
an efficiency or convergence statement: passing 4-MCSE on 3×400 draws says nothing about ESS, R̂, or bias on a
real model.

## 2. Coverage of the failure modes this run actually found

| defect found in this run | would WP8's test catch it? | why |
|---|---|---|
| Pathfinder metric applied to the **wrong coordinates** (the P0, fixed by `wp/wp1`) | **No** | the fixture is 3 parameters whose creation order equals sorted order, so the tuple→dict permutation is the identity permutation and cannot mis-assign anything |
| **Orientation** error (`inv(cov)` where the kernel needs the covariance) | **No** — and this is measured, not argued (see §3) | the fixture's variances are 1 / 4 / 0.25, a 16× spread; the failure needs a wide-spread target (the run's decisive fixture used 1e2 … 1e-2, a 10⁴ spread) for an inverted metric to collapse a coordinate inside tolerance |
| `imm_source` arms disagreeing on convention (`9f892b2`) | No | the test exercises one arm (`mc`) |
| ChEES trajectory-length rate limit (`wp/wp2`) | No | efficiency, not moments |
| Collapsed/failed adaptation (frozen ε ≈ 0, metric NaN) | **Yes** | mean or variance would blow past 4 MCSE, or the shape assert fires |

So the test earns its keep as a smoke test against catastrophic breakage, and it is a genuine improvement over
the previous suite (which had no end-to-end moment check at all). It must **not** be cited as evidence that the
metric path is statistically correct.

## 3. The decisive check — run the test on the broken tree

Method: `~/scratch/wp8_legacy_check.py`, which does `sys.path.insert(0, "/home/akbar/pf/legacy")`, asserts
`pfcheeshmc.__file__` starts with the legacy path, and runs **the wp8 branch's own test file** against it.

```
TREE: /home/akbar/pf/legacy/pfcheeshmc/__init__.py
.                                                    [100%]
1 passed in 14.83s
```

**The test passes on `ecd9ba1`** — the tree that carries the coordinate-ordering P0 and the failing default
orientation, i.e. the tree this entire run was convened to fix. A test that is green on the broken tree cannot
be the gate that keeps it from breaking again. (Same run, same forced-CPU device; the only difference is the
package under test.)

## 4. What would give it teeth

Two changes, both cheap and both inside the 60 s budget:

1. **Unsorted names.** Give the fixture parameters whose creation order differs from sorted order (e.g. create
   `z` before `a`), which makes the permutation non-identity. Then the ordering defect is detectable.
2. **Wide scale spread.** Use variances spanning ~10⁴ (say 1, 1e-2, 1e2) instead of 16×. The orientation defect
   then collapses the wide coordinate, exactly as in the 2×2 fixture.

`tests/test_metric_orientation.py` (17 tests, 12 failing on `ecd9ba1`) already covers both cases at the metric
level; the gap is only in the *end-to-end* test, which is the one a casual contributor runs.

## Disposition

**BLOCKING:** none for the merge — the plan's WP8 acceptance text asks for "an anisotropic Gaussian, mean and
variance within 4 MCSE (CPU ≤ 60 s)", and that is delivered and true.
**NON-BLOCKING (recorded so it is not over-cited):** the test passes on the broken pre-fix tree, so it is a
smoke test, not a regression gate for the run's two P0s. Add unsorted names and a 10⁴ variance spread when the
file is next touched; until then no acceptance claim in `results/REPORT.md` may rest on it.
**NIT:** the docstring's teeth claim should point at `test_metric_orientation.py` for the ordering/orientation
cases rather than implying this file covers them.

> **RECOMMEND: merge** — with the limitation stated in the SUMMARY, which is where a future reader will look.

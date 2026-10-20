"""The production-pool rows must say which pool they ran with.

Why this exists: `--pf-pool-cap` was added to `tools/r1_bias.py` only after the first
production-pool pass OOM'd, and the run's whole capacity conclusion (uncapped pool is 5.7 GB at
d=175; `4C` is verdict-neutral) rests on being able to tell capped rows from uncapped ones. The
flag's value is echoed into every row's `kwargs`, so a plumbing slip -- a cap that silently stays
None, or a capped row filed under the uncapped arm -- is a wrong number in a committed artifact
rather than a crash. Checked against the committed file, not a fixture.
"""
import json
import os

ROWS = os.path.join(os.path.dirname(__file__), os.pardir, "results", "runs", "T1__prodpool.jsonl")
CAPPED = "int:prodpool_cap4C"
UNCAPPED = "int:prodpool"


def _rows():
    if not os.path.exists(ROWS):
        return None
    with open(ROWS) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_pool_cap_is_recorded_and_matches_the_arm():
    rows = _rows()
    if rows is None:
        # The file is a committed run artifact, so its absence means it was deleted, not that the
        # test is optional. Fail rather than skip silently -- that is the zero-collect trap.
        raise AssertionError(f"{ROWS} missing: the production-pool rows are a committed artifact")

    scored = [r for r in rows if not r.get("error")]
    capped = [r for r in scored if r["arm"] == CAPPED]
    uncapped = [r for r in scored if r["arm"] == UNCAPPED]
    assert capped, f"no scored {CAPPED} rows -- the capped arm produced nothing to check"
    assert uncapped, f"no scored {UNCAPPED} rows -- the uncapped control produced nothing to check"

    for r in capped:
        pk = json.loads(r["kwargs"])["pathfinder_kwargs"]
        assert pk.get("pf_pool_cap") == 4096, f"{r['model']} seed={r['seed']}: cap {pk.get('pf_pool_cap')!r} != 4096"
        assert pk["draws"] == 4000, f"{r['model']}: capped rows must still request the production pool"
    for r in uncapped:
        pk = json.loads(r["kwargs"])["pathfinder_kwargs"]
        # Absent is correct here: the uncapped cells ran on the harness before the flag existed
        # (git_sha 557a0c0), so they have no pf_pool_cap key at all. Present-and-set is not.
        assert pk.get("pf_pool_cap") is None, (
            f"{r['model']} seed={r['seed']}: uncapped arm carries cap {pk['pf_pool_cap']!r}")
        assert pk["draws"] == 4000


def test_capped_arm_covers_the_models_the_uncapped_one_could_not():
    """The point of the cap was d >= 25. If those cells are absent the conclusion is untested."""
    rows = _rows()
    if rows is None:
        raise AssertionError(f"{ROWS} missing")
    capped_models = {r["model"] for r in rows if r["arm"] == CAPPED and not r.get("error")}
    assert {"radon_noncentered", "logit_1000x25"} <= capped_models, (
        f"capped arm covers {sorted(capped_models)}; radon (d=175) and logistic (d=25) are the "
        "models that OOM uncapped, so without them the capacity result is unmeasured")

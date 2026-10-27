"""Golden draws: the sampler's output for a fixed seed, at small shapes, is equal to the last bit to what `tests/data` holds.

The files were made by `tests/make_golden.py` from the tag `archive/pre-cleanup` and this tree's code, which gave the same
arrays; they replace the `git archive` proofs of the cleanup (each ran the old code in a child process, about eight minutes
of the suite). A case here is one `sample_pf_chees_hmc` call (or a few, or one stage method) at C=8, D=10, W=20:

  * the default path of each ladder model (8 Schools, GEV, logit 1000x25, radon non-centred);
  * `1 < pm_chains < chains` (a ragged cycle), `pm_chains = chains` and the PyMC start (`init_source="adapt_diag"`) on GEV,
    whose shared start sits on xi = 0, so the start repair runs in each of them; the repair alone, on its three branches
    (backtrack, restart, the search from the origin), and the starts a Pathfinder stage hands to the chains;
  * the widening of literals narrower than floatX (a rebuilt model finding the first one's lift in the memo; shaped float32
    and int8 literals);
  * chunked loops: several warmup, sampling and transform chunks (the draws do not depend on the chunk sizes, so this file
    is also what the default chunks give);
  * the warmup stop rule firing, the four `imm_source` values, the pool cap;
  * float32, each in a fresh process with ``floatX=float32``.

What is compared is every posterior and sample_stats array, the numeric attributes (``frozen_step_size``,
``frozen_L_nominal``, the gradient counts) and the resolved config, exactly (`np.array_equal`, dtype and shape included).
"""
import json
import pathlib

import numpy as np
import pytest

import make_golden

DATA = pathlib.Path(__file__).resolve().parent / "data"


def _same(want, got):
    assert sorted(want) == sorted(got), (sorted(set(want) ^ set(got)))
    different = []
    for key in sorted(want):
        a, b = want[key], got[key]
        if key.endswith("config"):
            ok = json.loads(str(a)) == json.loads(str(b))
        else:
            ok = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        if not ok:
            different.append(key)
    return different


@pytest.mark.parametrize("case", make_golden.CASES)
def test_the_draws_are_those_of_the_golden_file(case):
    z = np.load(DATA / f"golden_{case}.npz")
    want = {k: z[k] for k in z.files}
    floatx = make_golden._spec()[case][0]
    got = make_golden.compute(case) if floatx == "float64" else make_golden.compute_in_child(case)
    different = _same(want, got)
    assert not different, (
        f"{len(different)} of {len(want)} arrays of {case} differ from tests/data/golden_{case}.npz: {different[:6]}. "
        "If the change is meant to alter the draws, regenerate with `python tests/make_golden.py` (all cases, or name them) "
        "and say in the commit message why the draws change; if not, the change is a bug.")


def test_every_golden_file_has_a_case_and_every_case_a_file():
    files = {p.name[len("golden_"):-len(".npz")] for p in DATA.glob("golden_*.npz")}
    assert files == set(make_golden.CASES)
    assert sum(p.stat().st_size for p in DATA.glob("golden_*.npz")) < 1_000_000


def test_the_cases_are_not_trivially_equal():
    """The cases that differ in an option or a seed give different draws (a file is not the same run twice), and are finite."""
    load = lambda c: np.load(DATA / f"golden_{c}.npz")
    assert not np.array_equal(load("gev_pm_chains_3")["posterior/μ"], load("gev_pm_chains_C")["posterior/μ"])
    z = load("8schools_rebuilt")
    assert not np.array_equal(z["call0/posterior/mu"], z["call1/posterior/mu"])
    for case in make_golden.CASES:
        z = load(case)
        for k in z.files:
            if "/posterior/" in k or k.startswith("posterior/"):
                assert np.isfinite(z[k]).all(), (case, k)

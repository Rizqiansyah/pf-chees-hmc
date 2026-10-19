"""Permanent shape guard for the WP0.2 reference npz files (A8).

The 2026-10-19 review caught that the first reference build stored per-variable
mean/var/MCSE as scalars even for vector-valued variables: x.mean() / x.var()
reduce over ALL axes, and the MCSE helper collapsed the library's per-parameter
array to its first element. The effect was silent -- WP3 scores every parameter
of a variable against one pooled number, and theta is 8 of 10 parameters in 8
Schools and 170 of 175 in radon NC.

This test fails if any stat array's shape is not equal to its sample array's
shape minus the leading (chain, draw) axes, or if any sample array disagrees
with the metadata chains/draws. The SAME guard, reference_shape_problems,
runs inside make_reference before it saves, so generation also fails loudly;
this test is the guard on the committed reference artifacts and calls the
production function rather than re-implementing the comparison.

The npz files live outside git (~/pf-runs/references/ per A8). The first
review found that a missing directory made the fixture collect ZERO tests,
so `pixi run test` went all-green with the guard doing nothing: that trap is
closed by test_reference_files_present, which FAILS (not skips) when the
directory is absent unless REFERENCE_FILES_OPTIONAL=1 is set explicitly. The
teeth test builds a synthetic legacy-format file and asserts the shared guard
reports it red.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from make_reference import reference_shape_problems  # noqa: E402


def _reference_dir():
    return os.environ.get(
        "PF_REFERENCE_DIR", os.path.expanduser("~/pf-runs/references")
    )


def _npz_paths():
    d = _reference_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")
    )


@pytest.fixture(params=_npz_paths())
def npz_path(request):
    return request.param


def test_reference_files_present():
    """Close the zero-collection trap: a missing reference directory used to
    make the guard fixture collect nothing, so the suite went all-green with
    the guard doing nothing. This fails instead, naming the path and the
    command that regenerates the files. Environments that genuinely cannot
    hold the run artifacts set REFERENCE_FILES_OPTIONAL=1 -- that is an
    explicit opt-out, not directory absence."""
    d = _reference_dir()
    paths = _npz_paths()
    if not paths and os.environ.get("REFERENCE_FILES_OPTIONAL", "0") != "1":
        pytest.fail(
            f"no .npz files under {d}; the reference shape guard collected nothing. "
            "Regenerate with `pixi run --manifest-path <main checkout>/pixi.toml "
            "env PYTHONPATH=<this checkout> python tools/make_reference.py <model>` "
            "(models: 8schools, gev, logit_1000x25, radon_noncentered), or set "
            "REFERENCE_FILES_OPTIONAL=1 in environments without the run artifacts."
        )


def test_reference_npz_keeps_parameter_axis(npz_path):
    z = np.load(npz_path)
    arr = {k: z[k] for k in z.files if not k.startswith("meta__")}
    names = [k.split("__", 1)[1] for k in z.files if k.startswith(("x__", "u__"))]
    assert names, f"no sample arrays in {npz_path}"
    first = arr["u__" + names[0]] if f"u__{names[0]}" in arr else arr["x__" + names[0]]
    chains = int(z["meta__chains"]) if "meta__chains" in z.files else first.shape[0]
    draws = int(z["meta__draws"]) if "meta__draws" in z.files else first.shape[1]
    probs = reference_shape_problems(arr, chains, draws, names)
    assert not probs, (
        f"shape contract violated in {npz_path}:\n  " + "\n  ".join(probs)
        + "\n(the 2026-10-19 incident: see tools/make_reference.py docstring)"
    )


def test_legacy_shape_fails_the_guard():
    """Proof the shared guard catches the legacy bug, calling the production
    function rather than an inline copy of it: build a small array the old,
    wrong way (scalar stats next to a vector sample) and assert the guard
    reports it. Renaming STATS or breaking the sample lookup in
    reference_shape_problems makes this test red."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 100, 3))  # (chain, draw, param)
    arr = {
        "u__theta": x,
        "mean__theta": np.asarray(x.mean()),  # the legacy bug: scalar
        "var__theta": np.asarray(x.var(ddof=1)),
        "mcse_x__theta": np.asarray(0.01),
        "mcse_x2__theta": np.asarray(0.02),
    }
    probs = reference_shape_problems(arr, chains=8, draws=100, names=["theta"])
    assert probs, "the legacy collapse must be detected"
    assert any(p.startswith("mean__theta") for p in probs), probs
    # and a clean array passes the same function
    clean = {
        "u__theta": x,
        "mean__theta": x.mean(axis=(0, 1)),
        "var__theta": x.var(axis=(0, 1), ddof=1),
        "mcse_x__theta": np.ones(x.shape[2:]),
        "mcse_x2__theta": np.ones(x.shape[2:]),
    }
    assert reference_shape_problems(clean, chains=8, draws=100, names=["theta"]) == []

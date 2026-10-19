"""Permanent shape guard for the WP0.2 reference npz files (A8).

The 2026-10-19 review caught that the first reference build stored per-variable
mean/var/MCSE as scalars even for vector-valued variables: x.mean() / x.var()
reduce over ALL axes, and the MCSE helper collapsed the library's per-parameter
array to its first element. The effect was silent -- WP3 scores every parameter
of a variable against one pooled number, and theta is 8 of 10 parameters in 8
Schools and 170 of 175 in radon NC.

This test fails if any stat array's shape is not equal to its sample array's
shape minus the leading (chain, draw) axes. The same assertion runs inside
make_reference before it saves, so generation also fails loudly; this test is
the guard on the committed reference artifacts.

The npz files live outside git (~/pf-runs/references/ per A8), so the test
skips when the directory is absent. Override the directory with
PF_REFERENCE_DIR (used here to prove the test is red on legacy files).
"""

import os

import numpy as np
import pytest

STATS = ("mean", "var", "mcse_x", "mcse_x2")


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


def test_reference_npz_keeps_parameter_axis(npz_path):
    z = np.load(npz_path)
    samples = [k for k in z.files if k.startswith(("x__", "u__"))]
    assert samples, f"no sample arrays in {npz_path}"
    for s in samples:
        x = z[s]
        assert x.ndim >= 2, f"{s} in {npz_path} is not (chain, draw, ...)"
        var = s.split("__", 1)[1]
        for st in STATS:
            key = f"{st}__{var}"
            assert key in z.files, f"{key} missing from {npz_path}"
            got = np.asarray(z[key]).shape
            want = x.shape[2:]
            assert got == want, (
                f"{key} in {npz_path}: stat shape {got} != sample shape {want} "
                f"({x.shape} minus chain/draw). The parameter axis was reduced -- "
                "see the module docstring for the 2026-10-19 incident."
            )


def test_legacy_shape_fails_the_guard(tmp_path, monkeypatch):
    """Proof the guard catches the legacy bug: build a small npz the old,
    wrong way (scalar stats next to a vector sample) and assert the same
    check fails on it."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 100, 3))  # (chain, draw, param)
    z = {
        "u__theta": x,
        "mean__theta": np.asarray(x.mean()),        # the legacy bug: scalar
        "var__theta": np.asarray(x.var(ddof=1)),
        "mcse_x__theta": np.asarray(0.01),
        "mcse_x2__theta": np.asarray(0.02),
        "meta__model": np.asarray("synthetic"),
    }
    p = tmp_path / "synthetic_legacy.npz"
    np.savez(p, **z)

    # same assertions as test_reference_npz_keeps_parameter_axis, run directly
    # on the synthetic file so the test does not depend on the directory being
    # populated
    loaded = np.load(p)
    sx = loaded["u__theta"]
    with pytest.raises(AssertionError):
        assert np.asarray(loaded["mean__theta"]).shape == sx.shape[2:]

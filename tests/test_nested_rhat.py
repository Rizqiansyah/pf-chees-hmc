"""Nested R-hat (arviz_stats.rhat_nested) as the R1 diagnostic — property tests, CPU.

R1 is C ~ 1,024 chains x D ~ 10 draws. Split R-hat is useless there: plan §2b chopped ONE converged
32x2,000 run into 1024x39 / 512x20 / 512x78 / 128x313 and reproduced the repo's "degraded" values
1.050 / 1.090 / 1.024 / 1.006. Nested R-hat pools *within* superchains and compares *between* them,
so short chains stop being the problem. These tests pin that behaviour on the library call, which
A6 requires us to use instead of reimplementing.
"""

import numpy as np
import pytest

arviz_stats = pytest.importorskip("arviz_stats")


def _nested(x, superchain_ids):
    return float(np.asarray(arviz_stats.rhat_nested(
        x, superchain_ids=superchain_ids, chain_axis=0, draw_axis=1)).max())


def test_stationary_chains_are_near_one_whatever_the_superchain_split():
    """Independent stationary chains: nR-hat ~ 1 even at 10 draws per chain (where R-hat fails)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 10, 3))          # (chains, draws, dim)
    ids_even = [i // 8 for i in range(64)]    # 8 superchains x 8 chains
    ids_many = [i // 2 for i in range(64)]    # 32 superchains x 2 chains
    assert _nested(x, ids_even) < 1.05
    assert _nested(x, ids_many) < 1.05


def test_dispersed_superchains_are_flagged_loudly():
    """Superchains started in different regions must read clearly > 1 at the same D.

    This is the case R-hat-at-10-draws cannot separate from noise: at W=0 the two groups are
    genuinely different, and a diagnostic that stays ~1 here is not usable as a gate.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, size=(8, 40, 2))
    b = rng.normal(loc=3.0, size=(8, 40, 2))
    x = np.concatenate([a, b], axis=0)
    ids = [0] * 8 + [1] * 8
    assert _nested(x, ids) > 1.5


def test_within_superchain_dispersion_is_not_the_signal_being_measured():
    """Offsetting every superchain identically leaves between-superchain variance untouched.

    Records what nested R-hat does and does not see, so nobody reads a passing nR-hat as
    "the chains are centred". It compares superchain means to each other, not to zero.
    """
    rng = np.random.default_rng(2)
    x = rng.normal(size=(16, 40, 2))
    ids = [i // 4 for i in range(16)]
    base = _nested(x, ids)
    shifted = x + 5.0
    # rank-normalisation is exact-math shift-invariant; the residual is float32 rank bookkeeping.
    assert abs(_nested(shifted, ids) - base) < 1e-3


def test_superchain_ids_must_be_equal_sized():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 20, 1))
    with pytest.raises(Exception):
        _nested(x, [0, 0, 0, 1, 1, 2])

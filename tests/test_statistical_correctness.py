"""Statistical correctness: anisotropic Gaussian through the public API.

An anisotropic Gaussian: mean and variance within 4 MCSE (CPU, <= 60 s).

The full public ``sample_pf_chees_hmc`` pipeline (PyMC init -> finite-start
repair -> Pathfinder -> step-size tuning -> ChEES warmup -> dynamic-HMC
sampling) is run on a data-free 3-dim Gaussian, so the posterior is exactly
N(MU, diag(VAR)). For every parameter the posterior mean must be within 4
MCSE of the analytic mean and the posterior variance within 4 MCSE of the
analytic variance. MCSE is computed from the drawn chain with
``arviz_stats.mcse`` (autocorrelation-aware, split-chain), never from the
independence approximation.

Budget: chosen so the test runs in <= 60 s on CPU (measured ~15 s on forced
CPU of this box, GPU not required and not used). The budget is smaller than
the pipeline's public defaults (draws 1000 / warmup 2000) because this is a
sanity test, not a convergence study; 800 ChEES steps on a well-conditioned
3-dim Gaussian with a Pathfinder metric is ample for the 4-MCSE tolerance.

Seed 42 is fixed and recorded in the commit message of the WP8 SUMMARY.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pymc as pm

from pfcheeshmc import sample_pf_chees_hmc

SEED = 42
CHAINS = 4
DRAWS = 400
WARMUP = 800

MU = np.array([0.5, -1.0, 0.0])
SD = np.array([1.0, 2.0, 0.5])  # anisotropic: 4x spread in standard deviations
VAR = SD**2
N_PARAMS = int(MU.size)


def make_model():
    """No observations: the posterior is exactly the prior N(MU, diag(VAR))."""
    with pm.Model() as m:
        pm.Normal("x", mu=MU, sigma=SD, shape=N_PARAMS)
    return m


def check_anisotropic_gaussian(idata, mu, var):
    """Per-parameter mean/variance check. Returns a list of violation strings.

    Empty list == pass. The caller picks ``mu``/``var`` so the same function
    proves the test has teeth: with a perturbed analytic mean the violations
    must be non-empty.
    """
    from arviz_stats import mcse

    x = idata.posterior["x"].values  # (chain, draw, d)
    violations = []
    for i in range(x.shape[-1]):
        xi = x[:, :, i]  # (chain, draw)
        mcse_mean = float(mcse(xi))
        mean_hat = float(xi.mean())
        if abs(mean_hat - mu[i]) > 4 * mcse_mean:
            violations.append(
                f"x[{i}] mean {mean_hat:.5f} vs analytic {mu[i]:.5f}: "
                f"|offset| {abs(mean_hat - mu[i]):.5f} > 4*MCSE {4 * mcse_mean:.5f}"
            )
        # Variance via the second moment around the analytic mean:
        # E[(x-mu)^2] = Var(x) + (E[x]-mu)^2, so MCSE of (x-mu)^2 is the
        # MCSE of the variance estimator to O(1/n).
        y = (xi - mu[i]) ** 2
        mcse_var = float(mcse(y))
        var_hat = float(xi.var(ddof=1))
        if abs(var_hat - var[i]) > 4 * mcse_var:
            violations.append(
                f"x[{i}] var {var_hat:.5f} vs analytic {var[i]:.5f}: "
                f"|offset| {abs(var_hat - var[i]):.5f} > 4*MCSE {4 * mcse_var:.5f}"
            )
    return violations


def test_anisotropic_gaussian_mean_and_variance_within_4_mcse():
    idata = sample_pf_chees_hmc(
        make_model(),
        draws=DRAWS,
        chains=CHAINS,
        random_seed=SEED,
        progressbar=False,
        chees_kwargs={"draws": WARMUP},  # warmup is the ChEES stage's "draws"
    )
    assert idata.posterior["x"].shape == (CHAINS, DRAWS, N_PARAMS), (
        f"unexpected posterior shape {idata.posterior['x'].shape}"
    )
    violations = check_anisotropic_gaussian(idata, MU, VAR)
    assert not violations, "outside 4 MCSE: " + "; ".join(violations)

"""
Regression test for sample_stats construction in PfCheesHMC._to_inference_data.

Guards two defects found by review of the BlackJAX 1.7 migration:
  1. HMCInfo was walked recursively and every leaf was flattened to its last
     path segment. HMCInfo.momentum and HMCInfo.proposal (IntegratorState:
     position, momentum, logdensity, logdensity_grad) are all keyed by the
     model's own variable names, so four writers collided on one key and
     sample_stats['mu'] ended up holding a logdensity GRADIENT.
  2. lp came out named 'logdensity' (the rename map was keyed on full dotted
     paths), and the value was the trajectory endpoint's logdensity rather than
     the sampled state's -- wrong on every rejected transition.

Runs on CPU in a couple of seconds; no MCMC, no GPU.
"""
import numpy as np
import pymc as pm

from blackjax.mcmc.hmc import HMCInfo
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.dynamic_hmc import DynamicHMCState

from pfcheeshmc.sampler import PfCheesHMC

C, T, J = 2, 3, 4          # chains, draws, size of the array-valued parameter


def _stacked_model_and_inputs():
    """Synthetic scan-stacked states/infos with distinct sentinel values."""
    scalar = lambda v: np.full((T, C), v, dtype=float)      # noqa: E731
    vector = lambda v: np.full((T, C, J), v, dtype=float)   # noqa: E731

    sampled_position = {"mu": scalar(1.0), "theta": vector(1.0)}
    sampled_grad = {"mu": scalar(7.0), "theta": vector(7.0)}
    momentum = {"mu": scalar(5.0), "theta": vector(5.0)}
    proposal_position = {"mu": scalar(2.0), "theta": vector(2.0)}
    proposal_grad = {"mu": scalar(8.0), "theta": vector(8.0)}

    states = DynamicHMCState(
        position=sampled_position,
        logdensity=scalar(-1.5),        # the value lp MUST take
        logdensity_grad=sampled_grad,
        random_generator_arg=np.zeros((T, C)),
    )
    infos = HMCInfo(
        momentum=momentum,
        acceptance_rate=scalar(0.65),
        is_accepted=np.ones((T, C), dtype=bool),
        is_divergent=np.zeros((T, C), dtype=bool),
        energy=scalar(2.0),
        proposal=IntegratorState(
            position=proposal_position,
            momentum=momentum,
            logdensity=scalar(-2.5),    # endpoint lp; must NOT become `lp`
            logdensity_grad=proposal_grad,
        ),
        num_integration_steps=np.full((T, C), 3, dtype=np.int32),
    )

    with pm.Model() as model:
        pm.Normal("mu")
        pm.Normal("theta", shape=J)
    return model, states, infos


def test_sample_stats_is_not_polluted_and_exposes_lp():
    model, states, infos = _stacked_model_and_inputs()
    idata = PfCheesHMC(model, seed=0)._to_inference_data(states, infos)

    stats = idata.sample_stats
    assert set(stats.data_vars) == {
        "lp", "diverging", "acceptance_rate", "is_accepted", "energy",
        "num_integration_steps",
    }, f"unexpected sample_stats vars: {sorted(stats.data_vars)}"

    # claim 1: model parameters must never appear as diagnostics
    assert "mu" not in stats.data_vars
    assert "theta" not in stats.data_vars

    # claim 2: lp exists and is the SAMPLED state's logdensity, not the endpoint's
    assert np.allclose(stats["lp"].values, -1.5)
    assert not np.allclose(stats["lp"].values, -2.5)

    # axes are (chain, draw, ...) like every other ArviZ group
    assert stats["acceptance_rate"].shape == (C, T)
    assert stats["diverging"].dtype == np.bool_

    # posterior transform still runs and is unaffected
    assert set(idata.posterior.data_vars) == {"mu", "theta"}
    assert idata.posterior["mu"].shape == (C, T)


if __name__ == "__main__":
    test_sample_stats_is_not_polluted_and_exposes_lp()
    print("PASS: sample_stats clean, lp from sampled state")

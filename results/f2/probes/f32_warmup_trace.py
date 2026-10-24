"""F2k: the ChEES warmup trace (step size, harmonic acceptance, trajectory length) of one call, any floatX.

    python f32_warmup_trace.py TREE MODEL CHAINS WARMUP SEED
"""
import os, sys
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tools")]
import numpy as np, pytensor
import r1_bias
import pfcheeshmc.sampler as sm
from pfcheeshmc import sample_pf_chees_hmc

model, chains, warmup, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
real = sm.PfCheesHMC._run_chees
def spy(self, *a, **k):
    out = real(self, *a, **k)
    t = self._warmup_trace
    np.set_printoptions(precision=4, linewidth=200, suppress=False)
    print("floatX", pytensor.config.floatX)
    print("step_size           ", np.asarray(t.step_size))
    print("harmonic_acceptance ", np.asarray(t.harmonic_acceptance))
    print("trajectory_length   ", np.asarray(t.trajectory_length))
    print("divergences per step", np.asarray(out[2].is_divergent).sum(axis=1))
    return out
sm.PfCheesHMC._run_chees = spy
sample_pf_chees_hmc(r1_bias.build_model(model), draws=10, chains=chains, random_seed=seed, progressbar=False,
                    chees_kwargs={"draws": warmup})

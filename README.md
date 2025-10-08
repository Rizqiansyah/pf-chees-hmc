# pf-chees-hmc
A Pathfinder-ChEES-HMC sampler using BlackJAX backend, compatible with PyMC model.

# ChEES-HMC vs. NUTS (or Why Use ChEES-HMC?)
The No-U-Turn sampler (NUTS) is a common and ubiquotous general purpose posterior sampling method that is the default if you use a PyMC model with continuous priors and likelihood.
It's a variant of the Hamiltonian Monte Carlo (HMC), with a main advantage that it avoids the need to tune the step size ($\epsilon$) and trajectory length ($L$) parameters in HMC, which is notoriously difficult and time consuming to do.
However, NUTS require substantial overhead in its computation, and it is not suitable for GPU acceleration.

ChEES-HMC, as proposed in [1] tune the HMC parameter automatically during the initial warm up phase.
HMC itself is suitable for GPU acceleration, and has less computational overhead compared to NUTS.
This allows posterior sampling efficiently using GPU(s).

NUTS is usually run over a small number of long chains, usually with the number of chains equal to the number of CPU cores available.
In contrast, ChEES-HMC can be run over a lot of short chains simultaneously in a GPU (or multiple GPUs).
This means that ChEES-HMC has the potential to sample much faster as the chains are shorter (for the same amount of effective sample size compared to NUTS).

# Implementation
The Pathfinder-ChEES-HMC sampler follows these steps:
1. Initialize the starting point using `adapt_diag` from `pymc` (the default initialization for NUTS in PyMC)
2. Use the Pathfinder algorithm (with some jitter) to find a better initial starting point and mass matrix.
3. Perform short HMC runs to tune the $\epsilon$ parameter.
4. Perform ChEES adaptation to tune the $\epsilon$, $L$, initial starting point, and other HMC parameters.
5. Run a Dynamic HMC using the tuned parameters from ChEES (Step 5) and mass matrix from the Pathfinder algorithm (Step 2)

# Installation
Clone this repository, then use `pip install .`

# Use
See the `examples/` folder for some examples. In short
```
# Define your PyMC model here
with pymc.Model() as model:
  # Define prior ...
  # Define likelihood
  # Call the sampler
  trace = sample_pf_chees_hmc(draws=2000, chains=32)

#Postprocess the trace with arviz, as usual
```

For use with GPU sampling, see also the `examples/` folder.

# Common Issues
1. If the sampler is not converging, try increasing the number of ChEES draws using the argument
```
sample_pf_chees_hmc(chees_kwargs={"draws": 10_000})
```


# References
[1] Matthew Hoffman, Alexey Radul, Pavel Sountsov "An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo", in Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, PMLR 130:3907-3915, 2021. Available: https://proceedings.mlr.press/v130/hoffman21a.html

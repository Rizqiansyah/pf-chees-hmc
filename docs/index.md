# pf-chees-hmc

A Pathfinder-ChEES-HMC sampler for PyMC models, built on JAX and BlackJAX. `sample_pf_chees_hmc` takes a PyMC model with
continuous variables and returns an `arviz.InferenceData`. It runs about a thousand short chains in lock step, which is
what a GPU does well.

```python
import pymc as pm
from pfcheeshmc import sample_pf_chees_hmc

with pm.Model() as model:
    ...                                   # priors and likelihood
    idata = sample_pf_chees_hmc(draws=10, chains=1024, random_seed=42)
```

:::{warning}
Compiled Hessians on the GPU can be wrong. Set `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. The sampler itself uses
gradients only; see [GPU notes](gpu-notes.md).
:::

Source: <https://github.com/Rizqiansyah/pf-chees-hmc>. License: Apache-2.0.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
how-it-works
choosing-settings
metrics
rhat-many-chains
performance
case-study-gev-regression
compile-and-cache
float32
gpu-notes
troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
options
```

```{toctree}
:maxdepth: 1
:caption: Maintainers

design
upstream-drafts
```

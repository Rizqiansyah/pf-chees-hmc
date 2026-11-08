# Quick start

```python
import pymc as pm
from pfcheeshmc import sample_pf_chees_hmc

with pm.Model() as model:
    ...                                   # priors and likelihood
    idata = sample_pf_chees_hmc(draws=10, chains=1024, random_seed=42)
```

The model can also be passed as the first argument: `sample_pf_chees_hmc(model, draws=10, chains=1024)`.

- **Seed.** Pass `random_seed`. Without it the seed is the current date (`YYYYMMDD`), so every call on the same day uses the same seed.
- **Stage options** are forwarded as dicts: `pathfinder_kwargs`, `step_size_tune_kwargs`, `chees_kwargs`,
  `dynamic_hmc_kwargs`. A key a stage does not read raises a `TypeError` that names it and lists the valid ones.
- **Progress bars:** `progressbar=False` silences them.
- **Defaults:** `chains=32`, `draws=1000`, warmup `chees_kwargs={"draws": 2000}`. For the many-short-chain regime the
  sampler is designed for, see [Choosing chains, draws and warmup](choosing-settings.md).

## What comes back

`idata` is a normal `arviz.InferenceData`.

- `posterior` is in problem space, including `pm.Deterministic` variables.
- `sample_stats` has exactly `lp`, `diverging`, `acceptance_rate`, `is_accepted`, `energy` and `num_integration_steps`.
  `lp` is the log-density of the sampled state, as PyMC's JAX samplers emit it (`tests/test_inference_data.py` pins this).
- `idata.attrs` holds:
  - `config`: a JSON string of what every stage resolved (not only what was asked);
  - `frozen_step_size`, `frozen_L_nominal`: the step size and nominal trajectory length the draws used;
  - `frozen_inverse_mass_matrix` (flattened, in the kernel's coordinate order) and `frozen_inverse_mass_matrix_shape`;
  - `warmup_steps_used`, `warmup_grad_evals_per_chain`, `sampling_grad_evals_per_chain`.

## Examples

The notebooks in `examples/`:

- `8 Schools example.ipynb`
- `GEV Port Pirie example.ipynb`, with a NUTS comparison
- `New data every call.ipynb`: one model structure, new data and priors on every call
- `float32.ipynb`: where float32 helps and where it fails

Each opens with a `USE_FLOAT32` switch.

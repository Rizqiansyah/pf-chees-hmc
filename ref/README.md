# ref/ — source papers

Pulled 2026-10-17 for the methodology and performance review of this sampler. Files are the published PDFs, unmodified.

| File | Paper | Why it is here |
|---|---|---|
| `hoffman2021-chees-hmc.pdf` | Hoffman, Radul, Sountsov. *An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo.* AISTATS 2021, PMLR 130:3907-3915. https://proceedings.mlr.press/v130/hoffman21a.html | Defines the **ChEES criterion** and Algorithm 1, which the warmup in `pfcheeshmc/utils.py` implements. |
| `hoffman2021-chees-hmc-supp.pdf` | Supplementary material for the above. | Algorithm worked example, hyper-parameter details. |
| `zhang2022-pathfinder.pdf` | Zhang, Carpenter, Hoffman, Marcus, Sobie, Yao. *Pathfinder: Parallel quasi-Newton variational inference.* arXiv:2108.03782 (also JMLR 23(221):1-39, 2022). https://arxiv.org/abs/2108.03782 | Defines the quasi-Newton path approximation, the **PSIS-resampled draws (Algorithm 5: sample with replacement with probabilities proportional to w_j, "as an initialization algorithm for MCMC")** and the multi-start variant. Correction, verified against this PDF: the paper **never mentions a mass matrix**, so the PSIS-weighted mixture *covariance* in BlackJAX `pathfinder_adaptation` is a library extension, not a Pathfinder result. |
| `sountsov2022-snaper-chees-fast.pdf` | Sountsov, Hoffman. *Focusing on Difficult Directions for Learning HMC Trajectory Lengths.* arXiv:2110.11576. https://arxiv.org/abs/2110.11576 | Post-2021 successor work: **ChEES-fast**, **SNAPER**, and a turnkey multi-chain adaptive HMC that adapts step size, trajectory length and mass matrix together. Candidate for the "methodology update" question. |

## Implementation-side references (already installed, read them in place)

- `.pixi/envs/default/lib/python3.13/site-packages/blackjax/adaptation/chees_adaptation.py` — `base()` (what this repo calls) and the public `chees_adaptation()` warmup, which additionally offers Welford diagonal **mass-matrix estimation** (`mass_matrix_estimation`, `mass_matrix_window_fraction`), `decay_rate`, `LOG_UPDATE_CLIP`, and a private `_length_floor`.
- `.pixi/envs/default/lib/python3.13/site-packages/blackjax/vi/multipathfinder.py` and `blackjax/adaptation/pathfinder_adaptation.py` — `multi_approximate`, `psis_weights`, and `_psis_weighted_mixture_covariance`, i.e. the **paper-canonical** Pathfinder mass matrix, computed analytically as a dense `(d, d)` mixture covariance.
- `blackjax/adaptation/meads_adaptation.py`, `low_rank_adaptation.py` — the newer jointly-adaptive (step size + metric + trajectory length) families.

## License note

The arXiv PDFs are distributed under arXiv's non-exclusive licence; the PMLR paper under its stated licence. Keep them for reference only — they are not part of this package's distribution.

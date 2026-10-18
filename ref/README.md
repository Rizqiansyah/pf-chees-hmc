# ref/ — source papers

Pulled 2026-10-17 for the methodology and performance review of this sampler. Files are the published PDFs, unmodified.

| File | Paper | Why it is here |
|---|---|---|
| `hoffman2021-chees-hmc.pdf` | Hoffman, Radul, Sountsov. *An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo.* AISTATS 2021, PMLR 130:3907-3915. https://proceedings.mlr.press/v130/hoffman21a.html | Defines the **ChEES criterion** and Algorithm 1, which the warmup in `pfcheeshmc/utils.py` implements. |
| `hoffman2021-chees-hmc-supp.pdf` | Supplementary material for the above. | Algorithm worked example, hyper-parameter details. |
| `zhang2022-pathfinder.pdf` | **Zhang, Carpenter, Gelman, Vehtari.** *Pathfinder: Parallel quasi-Newton variational inference.* arXiv:2108.03782; **JMLR 23(306):1-49, 2022**. https://arxiv.org/abs/2108.03782 (author list and JMLR volume/article number verified against the arXiv metadata and jmlr.org on 2026-10-18; an earlier revision of this file listed Hoffman, Marcus, Sobie and Yao and JMLR 23(221) — both wrong.) | Defines the quasi-Newton path approximation, the **PSIS-resampled draws (Algorithm 5, "Pareto-smoothed importance resampling": sample with replacement with probabilities proportional to w_j, "in order to make it easy to use as an initialization algorithm for MCMC")** and the multi-start variant (Algorithm 2). Verified against this PDF: the phrases "mass matrix", "inverse mass", "mixture covariance" and "law of total variance" **do not occur in the paper**, so the PSIS-weighted mixture *covariance* used by BlackJAX `pathfinder_adaptation` is a library extension, not a Pathfinder result. The paper's only metric-related statement is a Conclusions remark that the **single** L-BFGS inverse Hessian is a reasonable initialisation for Stan's adaptation. Note also that Pathfinder itself warns the L-BFGS inverse Hessian "can fail to generate a positive definite inverse Hessian estimate", in which case "Pathfinder essentially fails". |
| `sountsov2022-snaper-chees-fast.pdf` | Sountsov, Hoffman. *Focusing on Difficult Directions for Learning HMC Trajectory Lengths.* arXiv:2110.11576. https://arxiv.org/abs/2110.11576 | Post-2021 successor work: **ChEES-fast**, **SNAPER**, and a turnkey multi-chain adaptive HMC that adapts step size, trajectory length and mass matrix together. Candidate for the "methodology update" question. |

## Implementation-side references (already installed, read them in place)

- `.pixi/envs/default/lib/python3.13/site-packages/blackjax/adaptation/chees_adaptation.py` — `base()` (what this repo calls) and the public `chees_adaptation()` warmup, which additionally offers Welford diagonal **mass-matrix estimation** (`mass_matrix_estimation`, `mass_matrix_window_fraction`), `decay_rate`, `LOG_UPDATE_CLIP`, and a private `_length_floor`.
- `.pixi/envs/default/lib/python3.13/site-packages/blackjax/vi/multipathfinder.py` and `blackjax/adaptation/pathfinder_adaptation.py` — `multi_approximate`, `psis_weights`, and `_psis_weighted_mixture_covariance`, i.e. the **paper-canonical** Pathfinder mass matrix, computed analytically as a dense `(d, d)` mixture covariance.
- `blackjax/adaptation/meads_adaptation.py`, `low_rank_adaptation.py` — the newer jointly-adaptive (step size + metric + trajectory length) families.

## License note

The arXiv PDFs are distributed under arXiv's non-exclusive licence; the PMLR paper under its stated licence. Keep them for reference only — they are not part of this package's distribution.

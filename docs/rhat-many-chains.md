# Reading R-hat with many chains

With about a thousand chains of a few hundred or thousand draws, R-hat goes to 1 only as each chain's own mean
converges, which a short chain cannot do. For m stationary chains of n draws that agree, with integrated autocorrelation
time tau, split R-hat is close to

```
sqrt((n - 1) / n + tau / n)          (n: draws per split chain, half the draws)
```

(Vats & Knudson 2021 write it with the ESS, m n / tau). So R-hat <= 1.01 needs tau < about n / 50 whatever the chains'
agreement: with 1,000 draws per chain, a coordinate whose autocorrelation time is above about 10 draws reads above 1.01
even when every chain is stationary.

An R-hat above 1.01 is then either slow mixing within stationary chains or chains that disagree. The ESS that ArviZ (and
Stan) report cannot tell the two apart: its estimate uses the between-chain variance, as R-hat does, so
`sqrt((n - 1) / n + 2 m / ESS)` matches the observed R-hat in both cases.

What does tell them apart is how R-hat falls with more draws. For stationary chains that agree, `R-hat^2 - (n - 1) / n`
is about `tau / n`, so it halves when the draws double; chains stuck apart keep their R-hat. Compare the R-hat of the first
half of the draws with that of all of them:

```python
import arviz as az
half = az.rhat(idata.posterior.isel(draw=slice(0, idata.posterior.sizes["draw"] // 2)))
full = az.rhat(idata.posterior)
```

Neither reading can see a bias that every chain shares (for example starts from one Pathfinder fit that a short warmup has
not left behind): chains that agree on a wrong answer read as converged. For that, compare with a reference run
(`benchmarks/bias.py`).

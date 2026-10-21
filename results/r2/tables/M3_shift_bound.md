# M3 worst mean shift in reference-sd units: bound sqrt(b2_max) (unconstrained space)
Exact when the argmax moment is (x); an upper bound when it is (x^2). Seeds 1-5.

| model | arm | per-seed sqrt(b2) [argmax] | max |
|---|---|---|---|
| 8schools | m3:ma | 0.013 [mu (x^2)] 0.018 [mu (x^2)] 0.017 [theta[5] (x^2)] 0.014 [theta[0] (x^2)] 0.014 [theta[3] (x^2)] | 0.018 |
| 8schools | m3:raw | 0.011 [mu (x^2)] 0.021 [mu (x^2)] 0.013 [theta[4] (x^2)] 0.019 [theta[3] (x^2)] 0.012 [tau_log__ (x)] | 0.021 |
| 8schools | m3:raw_refresh | 0.038 [mu (x^2)] 0.029 [mu (x^2)] 0.022 [tau_log__ (x^2)] 0.022 [theta[3] (x^2)] 0.022 [tau_log__ (x^2)] | 0.038 |
| gev | m3:ma | 0.009 [σ_log__ (x)] 0.029 [ξ_interval__ (x)] 0.013 [σ_log__ (x)] 0.010 [μ (x)] 0.010 [σ_log__ (x)] | 0.029 |
| gev | m3:raw | 0.017 [σ_log__ (x)] 0.025 [ξ_interval__ (x)] 0.015 [σ_log__ (x)] 0.013 [ξ_interval__ (x)] 0.011 [σ_log__ (x)] | 0.025 |
| gev | m3:raw_refresh | 0.022 [ξ_interval__ (x)] 0.018 [σ_log__ (x)] 0.025 [ξ_interval__ (x^2)] 0.010 [μ (x^2)] 0.038 [ξ_interval__ (x)] | 0.038 |
| logit_1000x25 | m3:ma | 0.021 [b[4] (x)] 0.020 [b[11] (x^2)] 0.020 [b[13] (x)] 0.013 [b[12] (x)] 0.019 [b[8] (x)] | 0.021 |
| logit_1000x25 | m3:raw | 0.020 [b[8] (x)] 0.026 [b[4] (x^2)] 0.016 [b[13] (x)] 0.015 [b[5] (x^2)] 0.023 [b[11] (x^2)] | 0.026 |
| logit_1000x25 | m3:raw_refresh | 0.022 [b[11] (x^2)] 0.025 [b[2] (x)] 0.020 [b[6] (x^2)] 0.022 [b[5] (x)] 0.039 [b[11] (x)] | 0.039 |
| radon_noncentered | m3:ma | 0.020 [beta_raw[30] (x^2)] 0.031 [sigma_beta_log__ (x^2)] 0.024 [beta_raw[4] (x^2)] 0.025 [beta_raw[36] (x)] 0.020 [beta_raw[4] (x^2)] | 0.031 |
| radon_noncentered | m3:raw | 0.029 [beta_raw[48] (x^2)] 0.022 [beta_raw[56] (x^2)] 0.024 [beta_raw[4] (x^2)] 0.021 [beta_raw[9] (x^2)] 0.024 [beta_raw[49] (x^2)] | 0.029 |
| radon_noncentered | m3:raw_refresh | 0.205 [alpha_raw[41] (x)] 0.028 [alpha_raw[33] (x^2)] 0.178 [beta_raw[76] (x)] 0.094 [sigma_beta_log__ (x^2)] 0.205 [alpha_raw[47] (x)] | 0.205 |

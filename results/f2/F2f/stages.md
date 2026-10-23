| model | main cold | main warm | main warm, new data | f2 cold | f2 warm | f2 warm, new data | warm speed-up |
|---|---|---|---|---|---|---|---|
| 8schools | 36.8 s | 6.03 s | 5.96 s | 17.2 s | 0.23 s | 0.22 s | 26.4x |
| gev | 46.7 s | 10.21 s | 10.27 s | 34.0 s | 0.55 s | 0.56 s | 18.6x |
| logit_1000x25 | 51.3 s | 8.17 s | 8.23 s | 31.6 s | 1.43 s | 1.50 s | 5.7x |
| radon_noncentered | 81.1 s | 20.90 s | 19.93 s | 45.2 s | 6.50 s | 4.87 s | 3.2x |

First call in a fresh process (s): no disk cache; disk cache warm with the same data; disk cache warm with NEW data

| model | main, no cache | main, cache, same data | main, cache, new data | f2, no cache | f2, cache, new data |
|---|---|---|---|---|---|
| 8schools | 36.8 (31.5 compile) | 6.4 (1.8 compile) | 10.7 (6.2 compile) | 17.2 (14.3 compile) | 3.7 (1.0 compile) |
| gev | 46.7 (39.1 compile) | 9.0 (2.2 compile) | 15.2 (8.5 compile) | 34.6 (28.9 compile) | 7.4 (2.0 compile) |
| logit_1000x25 | 51.3 (44.8 compile) | 7.7 (1.8 compile) | 11.5 (5.5 compile) | 31.9 (27.8 compile) | 4.8 (0.9 compile) |
| radon_noncentered | 81.1 (60.9 compile) | 25.9 (6.1 compile) | 24.9 (8.1 compile) | 43.4 (34.8 compile) | 9.8 (1.2 compile) |

Warm call per stage, mean of calls 2-3, seconds (same data)

| model | tree | wall | init | pathfinder | eps0 | warmup | sampling | transform | rest |
|---|---|---|---|---|---|---|---|---|---|
| 8schools | main | 6.03 | 0.53 | 1.13 | 1.02 | 1.67 | 1.52 | 0.01 | 0.16 |
| 8schools | f2 | 0.23 | 0.02 | 0.04 | 0.00 | 0.05 | 0.02 | 0.01 | 0.07 |
| gev | main | 10.21 | 1.70 | 1.83 | 1.47 | 2.78 | 2.02 | 0.06 | 0.36 |
| gev | f2 | 0.55 | 0.21 | 0.04 | 0.00 | 0.12 | 0.05 | 0.02 | 0.10 |
| logit_1000x25 | main | 8.17 | 0.49 | 1.37 | 1.15 | 3.15 | 1.76 | 0.00 | 0.24 |
| logit_1000x25 | f2 | 1.43 | 0.01 | 0.07 | 0.01 | 1.09 | 0.19 | 0.02 | 0.05 |
| radon_noncentered | main | 20.90 | 0.92 | 7.98 | 1.82 | 7.24 | 2.48 | 0.03 | 0.43 |
| radon_noncentered | f2 | 6.50 | 0.04 | 0.25 | 0.02 | 5.60 | 0.29 | 0.17 | 0.13 |

Cold call per stage (call 1, same data), seconds / XLA compiles

| model | tree | wall | init | pathfinder | eps0 | warmup | sampling | transform | rest |
|---|---|---|---|---|---|---|---|---|---|
| 8schools | main | 36.8 | 1.53 / 8 | 22.77 / 360 | 1.95 / 12 | 2.91 / 9 | 6.20 / 51 | 0.29 / 3 | 1.11 / 14 |
| 8schools | f2 | 17.2 | 0.58 / 10 | 9.33 / 96 | 1.44 / 1 | 3.14 / 7 | 1.67 / 3 | 0.10 / 1 | 0.94 / 16 |
| gev | main | 46.7 | 9.07 / 143 | 23.39 / 362 | 2.33 / 9 | 3.72 / 9 | 6.10 / 40 | 0.80 / 14 | 1.33 / 15 |
| gev | f2 | 34.0 | 8.98 / 164 | 13.31 / 77 | 2.98 / 1 | 4.52 / 4 | 2.41 / 1 | 0.50 / 1 | 1.26 / 15 |
| logit_1000x25 | main | 51.3 | 1.80 / 6 | 31.03 / 377 | 2.67 / 11 | 7.49 / 10 | 7.00 / 51 | 0.18 / 1 | 1.15 / 15 |
| logit_1000x25 | f2 | 31.6 | 0.67 / 7 | 17.33 / 78 | 3.25 / 1 | 7.54 / 7 | 1.88 / 3 | 0.05 / 1 | 0.87 / 15 |
| radon_noncentered | main | 81.1 | 2.08 / 9 | 46.11 / 498 | 2.77 / 12 | 16.36 / 10 | 11.93 / 51 | 0.58 / 10 | 1.27 / 15 |
| radon_noncentered | f2 | 45.2 | 0.85 / 10 | 18.07 / 101 | 5.38 / 1 | 16.74 / 7 | 2.77 / 3 | 0.31 / 1 | 1.09 / 16 |

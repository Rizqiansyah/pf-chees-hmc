"""Child process of tests/test_delayed_rejection.py::test_float32_invariance: the hard-edge invariance test of delayed
rejection with x64 off. Run with PYTENSOR_FLAGS=floatX=float32 (PyTensor sets jax_enable_x64 from floatX at import).
Prints one JSON line: x64, the draws' dtype and, per stage count, retries accepted, the KS p-value and the tail z.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import scipy.stats as st  # noqa: E402

import test_delayed_rejection as t  # noqa: E402  (imports pfcheeshmc, hence PyTensor, hence the x64 setting)
import jax  # noqa: E402
from pfcheeshmc import compiled  # noqa: E402

runs = {}
for stages, eps in ((2, t.EPS), (3, t.EPS3)):
    x, info = t._final_draws(compiled.sampling_chunk, 4, stages=stages, eps=eps)
    runs[str(stages)] = {"retry_accepted": int(np.sum(info.retry_accepted)),
                         "ks_p": float(st.kstest(x.astype(np.float64), t.EXACT.cdf).pvalue),
                         "tail_z": float(t._tail_z(x.astype(np.float64)))}
print(json.dumps({"x64": bool(jax.config.jax_enable_x64), "dtype": str(x.dtype), "runs": runs}))

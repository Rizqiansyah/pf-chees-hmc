#!/usr/bin/env python
"""R2-WP1 CPU smoke driver: run tools/r1_bias.py UNCHANGED and log the refresh values it drops.

tools/r1_bias.py rows carry frozen eps and frozen_L_nominal but not ``metric_refresh_T0`` or
``metric_refresh_lmax_whitened`` (the harness is R2-WP3's file, not this WP's). Those two live in
the ``params`` dict that ``PfCheesHMC._run_chees`` returns, so this driver wraps that one method,
appends the values to a sidecar JSONL named by $R2WP1_SIDECAR, and then hands over to
``r1_bias.main()`` with the caller's argv. Nothing in the sampler's numbers changes.

Usage (see SUMMARY.md for the exact commands):
  R2WP1_SIDECAR=<path> python results/r2/R2-WP1/smoke_driver.py <r1_bias.py flags>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS)

import numpy as np  # noqa: E402

SIDECAR = os.environ["R2WP1_SIDECAR"]


def _install():
    import pfcheeshmc.sampler as sm

    orig = sm.PfCheesHMC._run_chees

    def inner(self, *a, **k):
        out = orig(self, *a, **k)
        params = out[1]
        rec = {k_: (None if params.get(k_) is None else float(params[k_]))
               for k_ in ("metric_refresh_T0", "metric_refresh_lmax_whitened",
                          "metric_refresh_eps0", "metric_refresh_phaseA_end_eps",
                          "metric_refresh_eps_researched")
               if k_ in params}
        rec["has_refresh"] = "metric_refresh" in params
        rec["frozen_step_size"] = float(np.asarray(params["step_size"]).reshape(-1)[0])
        with open(SIDECAR, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return out

    sm.PfCheesHMC._run_chees = inner


if __name__ == "__main__":
    # r1_bias must be imported BEFORE pfcheeshmc/pytensor: it sets PYTENSOR_FLAGS (BLAS link flags)
    # at import, and a model with a matrix product (logit_1000x25) fails to compile without them.
    import r1_bias

    _install()

    sys.argv[0] = os.path.join(TOOLS, "r1_bias.py")
    r1_bias.main()

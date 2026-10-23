"""Child process of tests/test_compilation_cache.py: ONE sampler call in a fresh Python process.

    python compilation_cache_run.py TREE VIA CACHE_DIR DATA_SEED OUT.npz

VIA is how the cache is switched on: ``helper`` (``pfcheeshmc.enable_compilation_cache(CACHE_DIR)``), or
``none`` (the parent has put whatever it wants into the environment; this process touches nothing). The
model is ``tests/seeded_models.hier(DATA_SEED, DATA_SEED)``: other data AND other prior values at the same
shapes for another seed. Prints one JSON line: the JAX monitoring events this call produced, and the number
of files in CACHE_DIR before and after. The draws of ``mu`` and ``tau`` go to OUT.npz.
"""
import collections
import json
import os
import sys

tree, via, cache_dir, seed, out = sys.argv[1:6]
sys.path.insert(0, tree)
sys.path.insert(1, os.path.join(tree, "tests"))

import jax  # noqa: E402
import jax.monitoring  # noqa: E402
import numpy as np  # noqa: E402

events = collections.Counter()
seconds = collections.defaultdict(float)
jax.monitoring.register_event_listener(lambda event, **kw: events.update([event]))
jax.monitoring.register_event_duration_secs_listener(lambda event, secs, **kw: seconds.__setitem__(event, seconds[event] + secs))

import pfcheeshmc  # noqa: E402
from seeded_models import hier  # noqa: E402

if via == "helper":
    pfcheeshmc.enable_compilation_cache(cache_dir)


def n_files():
    return sum(len(files) for _, _, files in os.walk(cache_dir)) if os.path.isdir(cache_dir) else 0


before = n_files()
idata = pfcheeshmc.sample_pf_chees_hmc(
    hier(int(seed), int(seed)), draws=5, chains=16, random_seed=1, progressbar=False, chees_kwargs={"draws": 10})
np.savez(out, mu=idata.posterior["mu"].values, tau=idata.posterior["tau"].values)

short = lambda name: name.rsplit("/", 1)[-1]  # noqa: E731
print(json.dumps({
    "events": {short(k): v for k, v in events.items()},
    "seconds": {short(k): round(v, 3) for k, v in seconds.items()},
    "files_before": before,
    "files_after": n_files(),
}))

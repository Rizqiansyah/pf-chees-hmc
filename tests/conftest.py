"""Session-wide test setup: the on-disk compilation cache is off, and compiled programs are released per module.

The default cache (F2g)
    `sample_pf_chees_hmc` switches JAX's on-disk compilation cache on in the user's cache directory unless
    ``PFCHEESHMC_COMPILATION_CACHE`` is ``0``. Tests count compiles, and a cache hit is still a compile event
    (`tests/test_compilation_cache.py`), and they must not write into the home directory of whoever runs them. So the
    whole session opts out, here, before any test module is imported; a test of the default runs it in a child process
    with its own HOME and the variable removed.

Releasing compiled programs
    Every XLA executable a process holds costs it dozens of memory mappings, and Linux allows 65,530 per process
    (``vm.max_map_count``). The one-process suite ended at 61,028 on the F2 head f6108ac (counted from
    /proc/self/maps); a module that compiled more aborted it, several modules later, inside XLA ("Fatal Python
    error: Aborted" in ``backend_compile_and_load``). `release_compiled_programs` drops the executables, and what
    refers to them, when each test module is done. A module that needs programs compiled by another one compiles them
    again, which costs time (the suite takes about 13% longer) and nothing else: no test depends on it.
"""
import gc
import os

import jax
import pytest

os.environ["PFCHEESHMC_COMPILATION_CACHE"] = "0"


@pytest.fixture(scope="module", autouse=True)
def release_compiled_programs():
    yield
    gc.collect()
    jax.clear_caches()

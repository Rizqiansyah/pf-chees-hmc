"""Session-wide test setup: the on-disk compilation cache is off, and compiled programs are released when memory mappings run high.

The default cache (F2g)
    `sample_pf_chees_hmc` switches JAX's on-disk compilation cache on in the user's cache directory unless
    ``PFCHEESHMC_COMPILATION_CACHE`` is ``0``. Tests count compiles, and a cache hit is still a compile event
    (`tests/test_compilation_cache.py`), and they must not write into the home directory of whoever runs them. So the
    whole session opts out, here, before any test module is imported; a test of the default runs it in a child process
    with its own HOME and the variable removed.

Releasing compiled programs (F2g, then F2j)
    Every XLA executable a process holds costs it dozens of memory mappings, and Linux allows 65,530 per process
    (``vm.max_map_count``). The one-process suite ended at 61,028 on the F2 head f6108ac (counted from
    /proc/self/maps); a module that compiled more aborted it, several modules later, inside XLA ("Fatal Python
    error: Aborted" in ``backend_compile_and_load``). `release_compiled_programs` drops the executables, and what
    refers to them (``gc.collect()``, then ``jax.clear_caches()``), when a test module is done and the process holds
    more than `release_threshold(limit)` mappings; where /proc cannot be read it releases after every module.
    A module that needs programs compiled by an earlier one finds them while they are held, and compiles them again
    after a release, which costs time and nothing else: no test depends on either (a test that counts compiles in
    this process counts a second call, or a shape no other module uses).

    The threshold is ``limit // 2 - MODULE_GROWTH``: a module that starts at or under it ends under half the limit.
    ``MODULE_GROWTH`` is the largest growth of the count inside one test module, measured with
    ``results/f2/probes/mapcount_modules.py`` (each module alone in a fresh process, so from empty caches: its peak
    minus its start, 32 modules). The largest are test_pf_closed_form.py 12,589, test_compile_once.py 12,582,
    test_chunked_loops.py 8,584 and test_lift_memo.py 6,349; the constant is 16,000, 27% above the largest. At
    65,530 the threshold is 16,765. The rule is in terms of the limit read at run time, so a host with a lower limit
    releases sooner (at 32,768 and below, after every module, as before F2j).
    Measured on the suite (575 tests): peak PEAK mappings against 32,765 (half), wall WALL s against BASE s with
    the release after every module.
"""
import gc
import os

import jax
import pytest

os.environ["PFCHEESHMC_COMPILATION_CACHE"] = "0"

PEAK_FRACTION = 0.5       # of vm.max_map_count: where a test module may end
MODULE_GROWTH = 16_000    # mappings one test module may add (largest measured: 12,589)


def mapping_count():
    """The memory mappings this process holds (the lines of /proc/self/maps), or None where that cannot be read."""
    try:
        with open("/proc/self/maps", "rb") as f:
            return f.read().count(b"\n")
    except OSError:
        return None


def mapping_limit():
    """``vm.max_map_count``, or None where it cannot be read."""
    try:
        with open("/proc/sys/vm/max_map_count") as f:
            return int(f.read())
    except (OSError, ValueError):
        return None


def release_threshold(limit):
    """The count above which the programs are released after a test module."""
    return int(limit * PEAK_FRACTION) - MODULE_GROWTH


def should_release(count, limit):
    """Release when the count is above the threshold of the limit, and always when either is unknown."""
    return count is None or limit is None or count > release_threshold(limit)


def release_if_needed():
    if not should_release(mapping_count(), mapping_limit()):
        return False
    gc.collect()
    jax.clear_caches()
    return True


@pytest.fixture(scope="module", autouse=True)
def release_compiled_programs():
    yield
    release_if_needed()

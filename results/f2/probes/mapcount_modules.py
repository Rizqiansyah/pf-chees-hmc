"""Measurement only (F2j): memory mappings per test module, and the cost of reading them.

Per test module it records the /proc/self/maps line count
  start  before the module's first setup (before its fixtures),
  peak   the highest count seen inside the module (test boundaries, plus a sampler thread every
         MAPCOUNT_SAMPLE_S seconds, default 0.25; 0 = no thread),
  end    after its last test, before the module-scoped fixtures are finalised,
  after  after they are finalised (so `end - after` is what a release gave back),
and prints a table and a MAPCOUNT2 line at the end of the session. ``growth = peak - start`` is the number
`tests/conftest.py` keeps as margin.

Use (alone in the process, a module starts from empty caches):
    PYTHONPATH=results/f2/probes:$PWD python -m pytest tests/test_x.py -p mapcount_modules
Env:
    MAPCOUNT_MODULES_OUT=file   one JSON line per module (appended)
    MAPCOUNT_PER_TEST=1         also read the count after every test call (3,000+ reads; off by default, because
                                each read is a few ms on the test thread and the timing runs must not pay it)
    MAPCOUNT_FORCE=low|high     replace `conftest.release_threshold`: low = release after every module,
                                high = never (used for the must-fail runs of F2j)
Reading costs about 0.5 ms per 1,000 mappings, so the default sampler is cheap but not free.
"""
import json
import os
import sys
import threading
import time

import pytest

MODS = []
CUR = {"module": None, "start": 0, "peak": 0, "tests": 0, "node": ""}
GLOBAL = {"peak": 0, "node": "", "reads": 0}


def _n():
    with open("/proc/self/maps", "rb") as f:
        return f.read().count(b"\n")


def _see(n, node=""):
    GLOBAL["reads"] += 1
    if n > GLOBAL["peak"]:
        GLOBAL["peak"], GLOBAL["node"] = n, node or CUR["node"]
    if CUR["module"] is not None and n > CUR["peak"]:
        CUR["peak"] = n


def _sampler(period):
    while True:
        time.sleep(period)
        try:
            _see(_n())
        except OSError:
            return


def pytest_configure(config):
    period = float(os.environ.get("MAPCOUNT_SAMPLE_S", "0.25"))
    if period > 0:
        threading.Thread(target=_sampler, args=(period,), daemon=True).start()


def pytest_collection_finish(session):
    force = os.environ.get("MAPCOUNT_FORCE")
    if force:
        conftest = sys.modules["conftest"]
        conftest.release_threshold = (lambda limit: -1) if force == "low" else (lambda limit: 10 ** 12)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    module = item.nodeid.split("::")[0]
    if module != CUR["module"]:
        n = _n()
        CUR.update(module=module, start=n, peak=n, tests=0, node=item.nodeid)
        _see(n, item.nodeid)
    CUR["tests"] += 1
    CUR["node"] = item.nodeid


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield
    if os.environ.get("MAPCOUNT_PER_TEST"):
        _see(_n(), item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    last = nextitem is None or nextitem.nodeid.split("::")[0] != CUR["module"]
    end = _n() if last else None
    if end is not None:
        _see(end, item.nodeid)
    yield
    if last:
        after = _n()
        row = {"module": CUR["module"], "tests": CUR["tests"], "start": CUR["start"], "peak": CUR["peak"],
               "end": end, "after": after, "growth": CUR["peak"] - CUR["start"]}
        MODS.append(row)
        if os.environ.get("MAPCOUNT_MODULES_OUT"):
            with open(os.environ["MAPCOUNT_MODULES_OUT"], "a") as f:
                f.write(json.dumps(row) + "\n")


def pytest_sessionfinish(session, exitstatus):
    limit = open("/proc/sys/vm/max_map_count").read().strip()
    lines = [f"{'module':40s} {'tests':>5s} {'start':>7s} {'peak':>7s} {'end':>7s} {'after':>7s} {'growth':>7s}"]
    for r in MODS:
        lines.append(f"{r['module']:40s} {r['tests']:5d} {r['start']:7d} {r['peak']:7d} {r['end']:7d} "
                     f"{r['after']:7d} {r['growth']:7d}")
    lines.append(f"MAPCOUNT2 peak {GLOBAL['peak']} at {GLOBAL['node']}; reads {GLOBAL['reads']}; "
                 f"max growth {max((r['growth'] for r in MODS), default=0)}; "
                 f"releases {sum(1 for r in MODS if r['after'] < r['end'] - 3000)}; max_map_count {limit}")
    print("\n" + "\n".join(lines))

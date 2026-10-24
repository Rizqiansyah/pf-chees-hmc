"""Measurement only (F2g): peak /proc/self/maps line count of the pytest process, sampled after every test setup/call/teardown.

Use: PYTHONPATH=results/f2/probes:$PYTHONPATH python -m pytest tests -p mapcount_plugin; MAPCOUNT_OUT=file also writes the line.
"""
import os, pytest
PEAK = [0, ""]
def _n():
    with open("/proc/self/maps", "rb") as f:
        return f.read().count(b"\n")
def _sample(nodeid):
    n = _n()
    if n > PEAK[0]:
        PEAK[0], PEAK[1] = n, nodeid
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    yield
    _sample(item.nodeid)
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield
    _sample(item.nodeid + " (call)")
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    _sample(item.nodeid + " (teardown)")
def pytest_sessionfinish(session, exitstatus):
    line = (f"MAPCOUNT peak {PEAK[0]} at {PEAK[1]}; final {_n()}; max_map_count {open('/proc/sys/vm/max_map_count').read().strip()}")
    print("\n" + line)
    if os.environ.get("MAPCOUNT_OUT"):
        open(os.environ["MAPCOUNT_OUT"], "w").write(line + "\n")

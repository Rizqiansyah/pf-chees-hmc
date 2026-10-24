"""F2j: the suite releases compiled programs when its memory-mapping count is high, and after every module when it cannot tell.

What a wrong answer here would silently cost: a comparison off by one is harmless, but a rule that never releases
brings back the abort inside XLA at ``vm.max_map_count`` (the one-process suite reached 61,028 of 65,530 at f6108ac),
and one that always releases costs 13% of the suite's time. `tests/conftest.py` decides from two numbers, so the
decision is tested on numbers; the readers are tested with ``open`` replaced.
"""
import builtins
import os

import jax
import pytest

import conftest

LIMIT = 65_530


def test_threshold_is_half_the_limit_less_the_growth_of_one_module():
    assert conftest.release_threshold(LIMIT) == LIMIT // 2 - conftest.MODULE_GROWTH == 16_765
    assert conftest.release_threshold(2 * LIMIT) - conftest.release_threshold(LIMIT) == LIMIT // 2


def test_a_count_at_the_threshold_is_kept_and_one_above_it_is_released():
    t = conftest.release_threshold(LIMIT)
    assert [conftest.should_release(c, LIMIT) for c in (0, t - 1, t, t + 1, LIMIT)] == [False, False, False, True, True]


def test_where_the_count_or_the_limit_cannot_be_read_the_programs_are_released():
    assert conftest.should_release(None, LIMIT) is True
    assert conftest.should_release(1, None) is True
    assert conftest.should_release(None, None) is True


def test_a_limit_under_twice_the_growth_releases_after_every_module():
    limit = 2 * conftest.MODULE_GROWTH                  # the threshold is 0: any mapping is over it
    assert conftest.release_threshold(limit) == 0 and conftest.should_release(1, limit)


@pytest.mark.skipif(not os.path.exists("/proc/self/maps"), reason="no /proc")
def test_the_readers_see_this_process():
    with open("/proc/self/maps", "rb") as f:
        lines = len(f.read().splitlines())
    assert abs(conftest.mapping_count() - lines) < 100       # a Python process with JAX loaded holds thousands
    assert conftest.mapping_limit() == int(open("/proc/sys/vm/max_map_count").read())


def test_readers_give_none_where_proc_is_unreadable_or_not_a_number(monkeypatch):
    real_open = builtins.open

    def fail(path, *a, **k):
        if str(path).startswith("/proc/"):
            raise PermissionError(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fail)
    assert conftest.mapping_count() is None and conftest.mapping_limit() is None
    monkeypatch.undo()

    def garbage(path, *a, **k):
        return real_open(os.devnull if str(path) == "/proc/sys/vm/max_map_count" else path, *a, **k)

    monkeypatch.setattr(builtins, "open", garbage)
    assert conftest.mapping_limit() is None                 # an empty file is not a number


@pytest.mark.parametrize("count, released", [(100, False), (10 ** 6, True), (None, True)])
def test_the_fixture_collects_garbage_then_clears_the_caches_only_when_it_releases(monkeypatch, count, released):
    calls = []
    monkeypatch.setattr(conftest, "mapping_count", lambda: count)
    monkeypatch.setattr(conftest, "mapping_limit", lambda: LIMIT)
    monkeypatch.setattr(conftest.gc, "collect", lambda *a: calls.append("gc.collect"))
    monkeypatch.setattr(jax, "clear_caches", lambda: calls.append("jax.clear_caches"))
    assert conftest.release_if_needed() is released
    assert calls == (["gc.collect", "jax.clear_caches"] if released else [])

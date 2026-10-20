"""`summarise_rows.py smallest-pass` (R2-WP3a): synthetic row files, no sampler.

The rule it implements is plan §6's: a cell passes only if G-R1 reads `pass` on EVERY seed. Pooled rows
(G-R1-pool) are a diagnostic and must never decide it; unresolvable and ineligible cells are not passes;
and a model with no passing cell is reported as `none <= max W`, not as its largest W.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import summarise_rows as sr  # noqa: E402


def _r1(model="m", arm="A", W=100, seed=1, b2=0.001, floor=0.001, chains=1024, draws=10,
        gw=1000.0, gs=100.0, **kw):
    r = {"schema": "pf-r2/v1", "wp": "T", "arm": arm, "model": model, "chains": chains,
         "draws_per_chain": draws, "warmup": W, "seed": seed, "rep": 0, "cold": False,
         "b2_max": b2, "b2_floor": floor, "grads_per_chain_warmup": gw,
         "grads_per_chain_sampling": gs, "error": None, "pooled": False}
    r.update(kw)
    return r


def _pooled(**kw):
    return _r1(seed="pooled", pooled=True, chains=3072, **kw)


def _m5(model="m", tune=200, draws=100, seed=1, b2=0.001, floor=0.001, wall=10.0, core=300.0,
        grads=1500.0, **kw):
    r = {"schema": "pf-r2/v1", "step": "M5", "arm": "cpu:nuts", "model": model, "chains": 32,
         "tune": tune, "draws": draws, "seed": seed, "b2_max": b2, "b2_floor": floor,
         "wall_s": wall, "core_seconds": core, "grads_per_chain_total": grads, "error": None}
    r.update(kw)
    return r


def _one(rows, **kw):
    res, audit = sr.smallest_pass(rows, **kw)
    assert len(res) == 1
    return res[0], audit


# ---------------------------------------------------------------- R1 rows: smallest W

def test_smallest_w_must_pass_on_every_seed_and_reports_median_grads_and_per_seed_values():
    rows = [_r1(W=50, seed=1), _r1(W=50, seed=2, b2=0.09), _r1(W=50, seed=3),       # one seed fails
            _r1(W=100, seed=1, gw=900), _r1(W=100, seed=2, b2=0.004, gw=1000),
            _r1(W=100, seed=3, gw=1300),
            _r1(W=200, seed=1), _r1(W=200, seed=2), _r1(W=200, seed=3)]
    res, _ = _one(rows)
    assert res["pt"] == 100 and res["seeds"] == [1, 2, 3]
    assert res["grads"] == 1100.0                              # median of 1000, 1100, 1400
    assert res["per_seed"] == [(1, 0.001, 0.001), (2, 0.004, 0.001), (3, 0.001, 0.001)]
    assert {c["pt"]: c["state"] for c in res["cells"]} == {50: "fail", 100: "PASS", 200: "PASS"}


def test_pooled_rows_never_decide_and_are_reported_beside_the_answer():
    rows = [_r1(W=50, seed=1), _r1(W=50, seed=2, b2=0.09), _pooled(W=50, b2=0.0001),   # pooled passes
            _r1(W=100, seed=1), _r1(W=100, seed=2), _pooled(W=100, b2=0.5, floor=0.0003)]  # pooled fails
    res, _ = _one(rows)
    assert res["pt"] == 100                                    # W=50 not rescued, W=100 not blocked
    assert res["seeds"] == [1, 2]                              # "pooled" is not a seed
    assert res["pooled"] == (0.5, 0.0003)
    text = sr.format_smallest_pass([res], {})
    assert "pooled b2/floor: 0.5/0.0003 (diagnostic)" in text


def test_nothing_passing_prints_none_le_max_w():
    rows = [_r1(W=100, seed=1), _r1(W=100, seed=2, b2=0.3),
            _r1(W=400, seed=1, b2=0.2), _r1(W=400, seed=2)]
    res, audit = _one(rows)
    assert res["pt"] is None and res["max_pt"] == 400
    assert "none <= max W 400" in sr.format_smallest_pass([res], audit)
    capped, _ = _one(rows, max_w=100)
    assert "none <= max W 100" in sr.format_smallest_pass([capped], {})


def test_a_w_missing_a_seed_is_incomplete_not_a_pass():
    rows = [_r1(W=100, seed=1), _r1(W=100, seed=2),                     # seed 3 never ran at W=100
            _r1(W=200, seed=1), _r1(W=200, seed=2), _r1(W=200, seed=3)]
    res, _ = _one(rows)
    assert res["pt"] == 200
    assert res["cells"][0]["state"] == "incomplete (2/3 seeds)"
    res, _ = _one(rows[:2], seeds=[1, 2, 3])                            # expected seeds can be forced
    assert res["pt"] is None


def test_unresolvable_and_ineligible_cells_are_never_passes():
    rows = [_r1(W=100, seed=s, b2=0.0001, floor=0.02) for s in (1, 2)] \
        + [_r1(W=200, seed=s, b2=0.0001, chains=128) for s in (1, 2)] \
        + [_r1(W=300, seed=s) for s in (1, 2)]
    res, _ = _one(rows)
    states = {c["pt"]: c["state"] for c in res["cells"]}
    assert states[100] == "not resolvable" and states[200] == "not eligible"
    assert res["pt"] == 300


def test_error_void_starts_and_cold_rows_are_excluded_and_counted():
    rows = [_r1(W=100, seed=1), _r1(W=100, seed=2),
            _r1(W=100, seed=3, b2=None, error="OOM"),                   # error row
            _r1(W=100, seed=4, b2=0.0001, error="VOID: contended"),
            _r1(W=0, seed=1, arm="A"),                                  # pf_starts-style row
            _r1(W=50, seed=1, cold=True), _r1(W=50, seed=2, cold=True)]
    res, audit = _one(rows, warm_only=True)
    assert res["pt"] == 100 and res["seeds"] == [1, 2] and [c["pt"] for c in res["cells"]] == [100]
    assert audit["error"] == 1 and audit["VOID"] == 1 and audit["cold"] == 2


def test_arm_filter_and_last_row_wins_for_a_repeated_cell():
    rows = [_r1(arm="A", W=100, seed=1, b2=0.5), _r1(arm="A", W=100, seed=1),     # re-run replaces
            _r1(arm="A", W=100, seed=2), _r1(arm="B", W=100, seed=1, b2=0.5)]
    res, audit = _one(rows, arm="A")
    assert res["pt"] == 100 and res["arm"] == "A" and audit["cells with >1 row (last wins)"] == 1


def test_stored_verdict_that_disagrees_with_the_gate_is_audited():
    rows = [_r1(W=100, seed=1, g_r1="pass", b2=0.5), _r1(W=100, seed=2)]
    res, audit = _one(rows)
    assert res["pt"] is None                                   # the gate is recomputed, not trusted
    assert audit["stored g_r1 != recomputed"] == 1


# ---------------------------------------------------------------- M5 CPU rows: (tune, draws)

def test_m5_picks_the_cheapest_tune_plus_draws_passing_on_all_seeds():
    rows = []
    for tune, draws, ok, wall in [(200, 50, True, 3.0),      # C*D = 1600: ineligible whatever b2 is
                                  (200, 100, False, 5.0),    # seed 3 fails
                                  (500, 100, True, 9.0),     # cost 600
                                  (200, 400, True, 8.0),     # cost 600, cheaper wall -> wins the tie
                                  (1000, 400, True, 20.0)]:
        for seed in (1, 2, 3):
            bad = not ok and seed == 3
            rows.append(_m5(tune=tune, draws=draws, seed=seed, b2=0.3 if bad else 0.002,
                            wall=wall + seed, core=100.0 * (wall + seed), grads=1000.0 + seed))
    res, _ = _one(rows)
    assert res["mode"] == "m5" and res["pt"] == (200, 400)
    assert res["wall_s"] == 10.0 and res["core_seconds"] == 1000.0 and res["grads"] == 1002.0
    states = {c["pt"]: c["state"] for c in res["cells"]}
    assert states[(200, 50)] == "not eligible" and states[(200, 100)] == "fail"
    text = sr.format_smallest_pass([res], {})
    assert "tune=200 draws=400 (cost 600)" in text and "wall=10s core_s=1000" in text


def test_m5_none_passing_reports_the_largest_cost():
    rows = [_m5(tune=200, draws=50, seed=s) for s in (1, 2)] \
        + [_m5(tune=500, draws=100, seed=s, b2=0.4) for s in (1, 2)]
    res, _ = _one(rows)
    assert res["pt"] is None and res["max_cost"] == 600
    assert "none <= max cost 600" in sr.format_smallest_pass([res], {})


def test_m5_rows_written_by_the_driver_are_recognised_without_a_mode_flag():
    r = _m5()
    r.update(warmup=200, draws_per_chain=100)                  # pf-r1/v1 aliases the driver also writes
    assert sr._is_m5(r) and sr.smallest_pass([r], mode="auto")[0][0]["mode"] == "m5"


# ---------------------------------------------------------------- CLI

def test_cli_reads_a_glob_and_prints_the_answer(tmp_path, capsys):
    rows = [_r1(model="a", W=100, seed=1), _r1(model="a", W=100, seed=2),
            _r1(model="b", W=100, seed=1, b2=0.7), _r1(model="b", W=100, seed=2)]
    (tmp_path / "x1.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    (tmp_path / "x2.jsonl").write_text("\n".join(json.dumps(r) for r in rows[2:]) + "\n")
    assert sr.smallest_pass_main([str(tmp_path / "x*.jsonl"), "--arm", "A", "--cells"]) == 0
    out = capsys.readouterr().out
    assert "4 rows from 2 file(s)" in out
    assert "smallest pass W=100" in out and "none <= max W 100" in out
    with pytest.raises(FileNotFoundError):
        sr.smallest_pass_main([str(tmp_path / "nothing*.jsonl")])

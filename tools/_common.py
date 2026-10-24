"""Shared harness helpers for the pf-r1 run (implementation plan A5, A8).

Every harness must call `banner()` once and write rows with `write_row()` so that
the artifact contract in A8 holds. GPU work is serialised by `flock /tmp/pf-gpu.lock`;
run timed processes through `tools/gpu_run.sh`, which sets PF_GPU_LOCK=1, and timed
cells call `require_gpu_lock()` so a forgotten lock fails loudly instead of
producing a void timing (A5).
"""

import json
import os
import subprocess
import sys
import time

SCHEMA = "pf-r1/v1"
GPU_LOCK_FILE = "/tmp/pf-gpu.lock"

# A8 row schema: every field is always present, missing values are null.
_FIELDS = [
    ("schema", SCHEMA),
    ("wp", None),
    ("arm", None),
    ("git_sha", None),
    ("pfcheeshmc_file", None),
    ("model", None),
    ("d", None),
    ("chains", None),
    ("draws_per_chain", None),
    ("warmup", None),
    ("seed", None),
    ("rep", None),
    ("cold", None),
    ("kwargs", None),
    ("device", None),
    ("xla_flags", None),
    ("gpu_other_procs", None),
    ("cpu_contended", None),
    ("wall_s", None),
    ("stage_s", None),
    ("peak_mem_bytes", None),
    ("frozen_eps", None),
    ("frozen_L_nominal", None),
    ("steps_mean", None),
    ("grads_per_chain_warmup", None),
    ("grads_per_chain_sampling", None),
    ("pf_grads_total", None),
    ("divs_per_1k", None),
    ("warmup_divs", None),
    ("b2_max", None),
    ("b2_floor", None),
    ("b2_argmax", None),
    ("max_rhat", None),
    ("min_bulk_ess", None),
    ("min_tail_ess", None),
    ("ess_s_worst", None),
    ("grads_per_ess_worst", None),
    ("ebfmi_min", None),
    ("error", None),
]

_STAGE_KEYS = ["init", "pathfinder", "eps0", "warmup", "sampling", "postproc"]


def _git(args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=os.getcwd()
    ).stdout.strip()


def git_sha():
    return _git(["rev-parse", "--short", "HEAD"]) or "unknown"


def branch():
    return _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"


def pfcheeshmc_file():
    import pfcheeshmc

    return pfcheeshmc.__file__


def devices():
    import jax

    return [f"{d.device_kind}:{d.id}" for d in jax.devices()]


def xla_flags():
    return os.environ.get("XLA_FLAGS", "")


def gpu_other_procs():
    """Compute apps on the card other than this process tree (A5)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except Exception as exc:  # nvidia-smi missing or refused: record, do not crash
        return [f"query_failed:{exc}"]
    mine = {os.getpid(), os.getppid()}
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        if int(parts[0]) in mine:
            continue
        procs.append(":".join(parts[:2]))
    return procs


def peak_mem_bytes():
    """Peak device bytes. CPU returns None (memory_stats() is None there).

    IMPORTANT: this peak is CUMULATIVE FOR THE PROCESS — JAX 0.11.2 exposes no
    reset_memory_stats() on this device (verified on the A2000), so reading it after
    a cell reports the highest the process ever reached, not that cell's high-water
    mark. Stamp it with `set_peak_mem(row)` AFTER the cell, and for per-cell
    attribution run one process per cell (which A5 already wants for cold/warm
    separation). Never rely on the value `new_row()` auto-fills for a cell decision.
    """
    import jax

    try:
        return int(jax.devices()[0].memory_stats()["peak_bytes_in_use"])
    except Exception:
        return None


def set_peak_mem(row):
    """Stamp row['peak_mem_bytes'] with the device peak measured NOW (call after the cell)."""
    row["peak_mem_bytes"] = peak_mem_bytes()
    return row


def require_gpu_lock(what="timed GPU run"):
    """A5: timed GPU work must run inside flock /tmp/pf-gpu.lock (via gpu_run.sh)."""
    if os.environ.get("PF_GPU_LOCK") != "1":
        raise RuntimeError(
            f"{what} must run under the GPU lock: tools/gpu_run.sh <cmd> "
            f"(or flock {GPU_LOCK_FILE} ...)"
        )


def banner(model=None, cold=None, wp=None, arm=None):
    """Print the A8 provenance block and return it for embedding in a row."""
    info = {
        "git_sha": git_sha(),
        "branch": branch(),
        "pfcheeshmc_file": pfcheeshmc_file(),
        "device": devices(),
        "xla_flags": xla_flags(),
        "gpu_lock": os.environ.get("PF_GPU_LOCK") == "1",
        # a "cold" run that reads an on-disk compilation cache is a warm one (the library's default cache, F2g)
        "compilation_cache": {k: os.environ.get(k) for k in ("PFCHEESHMC_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR")},
        "cold": cold,
        "model": model,
        "wp": wp,
        "arm": arm,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print("[banner] " + json.dumps(info), flush=True)
    return info


def new_row(**kw):
    row = {k: (default() if callable(default := d) else d) for k, d in _FIELDS}
    for k, v in kw.items():
        if k not in row:
            raise KeyError(f"field {k!r} is not in the pf-r1/v1 schema")
        row[k] = v
    if row["pfcheeshmc_file"] is None:
        row["pfcheeshmc_file"] = pfcheeshmc_file()
    if row["git_sha"] is None:
        row["git_sha"] = git_sha()
    if row["device"] is None:
        row["device"] = devices()
    if row["xla_flags"] is None:
        row["xla_flags"] = xla_flags()
    if row["peak_mem_bytes"] is None:
        row["peak_mem_bytes"] = peak_mem_bytes()
    return row


def stages(**kw):
    """stage_s dict with all six A8 keys present."""
    s = {k: None for k in _STAGE_KEYS}
    for k, v in kw.items():
        if k not in s:
            raise KeyError(f"stage {k!r} is not in {os.linesep.join(_STAGE_KEYS)}")
        s[k] = v
    return s


def write_row(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(f"[row] {path} :: wp={row['wp']} arm={row['arm']} model={row['model']} "
          f"C={row['chains']} D={row['draws_per_chain']} W={row['warmup']} "
          f"seed={row['seed']} rep={row['rep']} wall={row['wall_s']} "
          f"error={row['error']}", flush=True)


def env_json(path="results/env.json", extra=None):
    """WP0.3: the environment record that every row is interpreted against."""
    import jax
    import jaxlib
    import blackjax
    import numpy
    import pymc
    import arviz
    import scipy
    import xarray

    try:
        import arviz_stats

        avs = getattr(arviz_stats, "__version__", "installed")
    except ImportError:
        avs = None

    env = {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "blackjax": blackjax.__version__,
        "pymc": pymc.__version__,
        "arviz": arviz.__version__,
        "arviz_stats": avs,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "xarray": xarray.__version__,
        "git_sha": git_sha(),
        "branch": branch(),
        "devices": devices(),
        "XLA_FLAGS": xla_flags(),
        "activation_env": {
            k: v for k, v in os.environ.items() if k.startswith(("XLA_", "PYTENSOR"))
        },
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "gpu_lock_file": GPU_LOCK_FILE,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        env.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(env, fh, indent=2)
    print(f"[env] wrote {path}")
    return env


def reference_path(model):
    return os.path.expanduser(f"~/pf-runs/references/{model}.npz")


def profile_dir():
    p = os.path.expanduser("~/pf-runs/profiles")
    os.makedirs(p, exist_ok=True)
    return p

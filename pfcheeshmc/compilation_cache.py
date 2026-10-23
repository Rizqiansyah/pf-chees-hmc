"""Opt-in on-disk cache for the XLA programs the sampler compiles (F2e).

A sampler call compiles nothing new within a Python process once the programs for its model structure exist
(F2b); the first call in a NEW process compiles them all again. JAX can keep compiled programs in a directory
and read them back in the next process. It is off unless a directory is set. There are two ways to set it:

* no code: the environment variables JAX reads for its own settings,
  ``JAX_COMPILATION_CACHE_DIR=/some/dir JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1``;
* `enable_compilation_cache`, which sets the same three in one call.

The two thresholds are what the helper adds. JAX writes a program only if it took at least 1 s to compile
(``jax_persistent_cache_min_compile_time_secs``), and nearly every program of a sampler call takes less:
8 Schools makes 135 compile requests in a first call and only 2 of them reach 1 s, so the directory alone
recovers about a third of the compile time while the three settings recover almost all of it
(see the README, "Compile once").
"""
from __future__ import annotations

import os

import jax

__all__ = ["enable_compilation_cache"]


def enable_compilation_cache(path: str | os.PathLike) -> None:
    """Keep the compiled XLA programs in the directory ``path``, and reuse them in the next Python process.

    It sets three JAX settings and nothing else:

    ``jax_compilation_cache_dir = path``
        Where the programs go; JAX creates the directory.
    ``jax_persistent_cache_min_compile_time_secs = 0``
        JAX's default of 1 s would skip nearly every program of a sampler call, as most compile faster.
    ``jax_persistent_cache_min_entry_size_bytes = -1``
        "No minimum, and no override" in JAX's wording. The default (0) lets JAX pick a platform-specific
        minimum; -1 does not.

    Call it once, near the top of the script, before the first sampler call. A directory that is already
    in use in this process (a program was compiled with one set) stays in use: JAX opens it once.

    What it covers: the XLA compilation of every program of the sampler (the three loops, the Pathfinder fit, the
    step-size search, the log-density, ...), on CPU and GPU. What it does not: JAX still traces each function and
    lowers it to StableHLO in every new process (about 2 s for a small model), and PyMC/PyTensor still build their
    graphs (about 0.1-0.4 s per call); neither is cached.

    The entries are keyed by the lowered program, the jaxlib version, the backend and its devices, and the XLA
    flags, so a different JAX, GPU or device count misses instead of misbehaving. Entries hold no data and no prior
    values: the sampler passes those at run time (F2a), so a new dataset or new priors at the same shapes use the
    entries of the old one. Deleting the directory is always safe; the next call writes it again. A cache is
    trusted code: use a directory only you can write to, and do not share it with users you do not trust.

    The same effect without any code: set ``JAX_COMPILATION_CACHE_DIR``,
    ``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`` and ``JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1`` in the
    environment (the directory alone is not enough, see above).
    """
    jax.config.update("jax_compilation_cache_dir", os.fspath(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

"""On-disk cache for the XLA programs the sampler compiles: on by default, in a per-user directory (F2e, F2g).

A sampler call compiles nothing new within a Python process once the programs for its model structure exist
(F2b); the first call in a NEW process compiles them all again. JAX can keep compiled programs in a directory
and read them back in the next process. `sample_pf_chees_hmc` switches that on, the way PyTensor keeps its
compiled C code in a per-user directory (``~/.pytensor``) by default:

* **Where**: `default_compilation_cache_dir`, ``$XDG_CACHE_HOME/pfcheeshmc/jax`` or ``~/.cache/pfcheeshmc/jax``.
* **When**: at the start of each `sample_pf_chees_hmc` call, by `ensure_default_compilation_cache`, which does
  nothing once it has run. Importing the package changes no JAX setting.
* **Never over a choice of yours**: if a JAX compilation cache directory is set already (``JAX_COMPILATION_CACHE_DIR``,
  ``jax.config``, ``jax.experimental.compilation_cache.set_cache_dir``), or ``jax_enable_compilation_cache`` is
  off, no JAX cache setting is touched.
* **Off**: set the environment variable ``PFCHEESHMC_COMPILATION_CACHE=0`` (also ``false``, ``no`` or ``off``)
  before the sampler call.
* **Explicit**: `enable_compilation_cache(path)` uses a directory of your choosing, whatever the variable says.

The two thresholds are what makes the cache worth having. JAX writes a program only if it took at least 1 s to
compile (``jax_persistent_cache_min_compile_time_secs``), and nearly every program of a sampler call takes less:
8 Schools at 1,024 chains makes 134 compile requests in a first call and only 3 of them reach 1 s, so the
directory alone takes a fresh process's first call from 13.6 s to 8.2 s, while all three settings take it to
3.4 s (CPU; see the README, "Keeping the compiled programs between Python processes").
"""
from __future__ import annotations

import os
import pathlib
import warnings

import jax

__all__ = ["default_compilation_cache_dir", "enable_compilation_cache"]

OPT_OUT_ENV = "PFCHEESHMC_COMPILATION_CACHE"
_OFF = ("0", "false", "no", "off")
# JAX's own defaults of the two thresholds (`jax/_src/config.py`); the default cache moves a threshold only from there
_JAX_DEFAULT_MIN_COMPILE_SECS, _JAX_DEFAULT_MIN_ENTRY_BYTES = 1.0, 0
_default_dir_in_use: str | None = None


def default_compilation_cache_dir() -> pathlib.Path:
    """``$XDG_CACHE_HOME/pfcheeshmc/jax``; ``~/.cache/pfcheeshmc/jax`` if ``XDG_CACHE_HOME`` is unset, empty or relative.

    The XDG base-directory convention puts a program's disposable files under the user's cache directory, as pip
    does (``~/.cache/pip``). Everything in it can be deleted: the next call writes it again.
    """
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    base = pathlib.Path(xdg) if os.path.isabs(xdg) else pathlib.Path.home() / ".cache"
    return base / "pfcheeshmc" / "jax"


def _refuse(path: pathlib.Path, why: str) -> bool:
    warnings.warn(f"pfcheeshmc: the compilation cache is off because {path} {why}. "
                  f"Set JAX_COMPILATION_CACHE_DIR to a directory of yours, or {OPT_OUT_ENV}=0 to silence this.",
                  RuntimeWarning, stacklevel=3)
    return False


def ensure_default_compilation_cache() -> bool:
    """Switch on the default on-disk cache for this process unless it is opted out of or already configured.

    Called at the start of every `sample_pf_chees_hmc`; idempotent. Returns whether the default directory is the
    one JAX uses afterwards. It sets ``jax_compilation_cache_dir`` and, if they are still at JAX's defaults,
    ``jax_persistent_cache_min_compile_time_secs = 0`` and ``jax_persistent_cache_min_entry_size_bytes = -1``
    (see `enable_compilation_cache`); nothing else, and nothing at all in the cases below.

    * ``PFCHEESHMC_COMPILATION_CACHE`` is ``0``, ``false``, ``no`` or ``off``: nothing.
    * A JAX cache directory is set already, or ``jax_enable_compilation_cache`` is off: nothing. The user's
      settings (size limit, thresholds, ...) are theirs.
    * The directory is created owner-only (0700). A cache is trusted code: if it exists and belongs to someone
      else or can be written by group or others, or cannot be created, the cache stays off and a
      `RuntimeWarning` says why. A failure to cache never fails a sampler call.

    The directory is set after other programs may have been compiled in this process: JAX decides whether the
    cache is in use at the first compile from ``jax_enable_compilation_cache`` and the platform, not from the
    directory, and looks for the directory again at every compile until it finds one
    (`jax/_src/compilation_cache.py`, ``is_cache_used`` and ``_initialize_cache``). Tested.
    """
    global _default_dir_in_use
    if os.environ.get(OPT_OUT_ENV, "").strip().lower() in _OFF:
        return False
    current = jax.config.jax_compilation_cache_dir
    if current is not None:
        return current == _default_dir_in_use
    if not jax.config.jax_enable_compilation_cache:
        return False
    path = default_compilation_cache_dir()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.mkdir(mode=0o700, exist_ok=True)
        info = path.stat()
    except OSError as e:
        return _refuse(path, f"cannot be created ({e.__class__.__name__}: {e})")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return _refuse(path, "belongs to another user")
    if info.st_mode & 0o022:
        return _refuse(path, f"can be written by group or others (mode {info.st_mode & 0o777:03o})")
    jax.config.update("jax_compilation_cache_dir", str(path))
    if jax.config.jax_persistent_cache_min_compile_time_secs == _JAX_DEFAULT_MIN_COMPILE_SECS:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    if jax.config.jax_persistent_cache_min_entry_size_bytes == _JAX_DEFAULT_MIN_ENTRY_BYTES:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    _default_dir_in_use = str(path)
    return True


def enable_compilation_cache(path: str | os.PathLike) -> None:
    """Keep the compiled XLA programs in the directory ``path``, and reuse them in the next Python process.

    The explicit choice: it works whatever ``PFCHEESHMC_COMPILATION_CACHE`` says, and it replaces the default
    directory if a sampler call has switched that on already. It sets three JAX settings and nothing else:

    ``jax_compilation_cache_dir = path``
        Where the programs go; JAX creates the directory.
    ``jax_persistent_cache_min_compile_time_secs = 0``
        JAX's default of 1 s would skip nearly every program of a sampler call, as most compile faster.
    ``jax_persistent_cache_min_entry_size_bytes = -1``
        "No minimum, and no override" in JAX's wording. The default (0) lets JAX pick a platform-specific
        minimum; -1 does not.

    JAX opens its cache directory once per process. If another one is in use (the default, or one set before),
    ``reset_cache`` makes JAX open ``path`` at the next compile instead.

    What it covers: the XLA compilation of every program of the sampler (the three loops, the Pathfinder fit, the
    step-size search, the log-density, ...), on CPU and GPU. What it does not: JAX still traces each function and
    lowers it to StableHLO in every new process (about 2 s for 8 Schools), and PyMC/PyTensor still build their
    graphs (0.05-0.14 s per call on the example models); neither is cached.

    The entries are keyed by the lowered program, the jaxlib version, the backend and its devices, and the XLA
    flags, so a different JAX, GPU or device count misses instead of misbehaving. Entries hold no data and no prior
    values: the sampler passes those at run time (F2a), so a new dataset or new priors at the same shapes use the
    entries of the old one. Deleting the directory is always safe; the next call writes it again. A cache is
    trusted code: use a directory only you can write to, and do not share it with users you do not trust.

    The same effect without any code: set ``JAX_COMPILATION_CACHE_DIR``,
    ``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`` and ``JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1`` in the
    environment (the directory alone is not enough, see above). A directory set that way is respected by the
    default, which then touches nothing.
    """
    path = os.fspath(path)
    in_use = jax.config.jax_compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    if in_use not in (None, path):
        from jax.experimental.compilation_cache import compilation_cache
        compilation_cache.reset_cache()

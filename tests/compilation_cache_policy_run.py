"""Child process of tests/test_compilation_cache.py: the rules of the default cache, one scenario after another.

    python compilation_cache_policy_run.py TREE SCRATCH_DIR

Each scenario starts from JAX's own defaults (every cache setting reset, JAX's cache forgotten) and a fresh HOME
under SCRATCH_DIR, calls `ensure_default_compilation_cache` (or the helper), and reports what it did. No sampler
runs here: the programs compiled are plain `jax.jit` functions, so the whole file takes a few seconds. Prints one
JSON line: scenario -> result.
"""
import collections
import json
import os
import pathlib
import stat
import sys
import warnings

tree, scratch = sys.argv[1:3]
sys.path.insert(0, tree)
scratch = pathlib.Path(scratch)

import jax  # noqa: E402
import jax.monitoring  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental.compilation_cache import compilation_cache as jax_cache  # noqa: E402

import pfcheeshmc  # noqa: E402
from pfcheeshmc import compilation_cache as cc  # noqa: E402

DEFAULTS = {"jax_compilation_cache_dir": None, "jax_persistent_cache_min_compile_time_secs": 1.0,
            "jax_persistent_cache_min_entry_size_bytes": 0, "jax_compilation_cache_max_size": -1,
            "jax_enable_compilation_cache": True}
events = collections.Counter()
jax.monitoring.register_event_listener(lambda event, **kw: events.update([event.rsplit("/", 1)[-1]]))
counter = [0]


def fresh(name, **env):
    """JAX's defaults, an empty HOME of its own, and exactly the environment variables given."""
    for key, value in DEFAULTS.items():
        jax.config.update(key, value)
    jax_cache.reset_cache()
    cc._default_dir_in_use = None
    home = scratch / name
    home.mkdir()
    for var in ("XDG_CACHE_HOME", cc.OPT_OUT_ENV):
        os.environ.pop(var, None)
    os.environ["HOME"] = str(home)
    os.environ.update(env)
    events.clear()
    return home


def settings():
    return {k: jax.config.values[k] for k in DEFAULTS}


def new_program():
    """Compile a program no earlier scenario compiled."""
    counter[0] += 1
    jax.jit(lambda x, k=counter[0]: x * (k + 1.5) + k)(jnp.arange(3.0)).block_until_ready()


def files(path):
    return sorted(p.name for p in pathlib.Path(path).glob("*-cache")) if os.path.isdir(path) else []


def call():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        on = cc.ensure_default_compilation_cache()
    return on, [f"{w.category.__name__}: {w.message}" for w in caught]


out = {}

home = fresh("fresh")
default = home / ".cache" / "pfcheeshmc" / "jax"
on, warned = call()
first = settings()
mode = {"leaf": stat.S_IMODE(default.stat().st_mode), "parent": stat.S_IMODE(default.parent.stat().st_mode)}
on_again, warned_again = call()
out["fresh"] = {"on": on, "warned": warned, "settings": first, "dir": str(default), "mode": mode,
                "again": {"on": on_again, "warned": warned_again, "same_settings": settings() == first},
                "function_dir": str(cc.default_compilation_cache_dir())}

for value in ("0", "false", "No", " OFF "):
    home = fresh(f"optout_{value.strip()}", **{cc.OPT_OUT_ENV: value})
    on, warned = call()
    out[f"optout[{value}]"] = {"on": on, "warned": warned, "settings": settings(), "home_has_cache": (home / ".cache" / "pfcheeshmc").exists()}
for value in ("1", "", "yes"):
    fresh(f"optin_{value or 'empty'}", **{cc.OPT_OUT_ENV: value})
    out[f"not_optout[{value}]"] = {"on": call()[0]}

home = fresh("user_dir")
jax.config.update("jax_compilation_cache_dir", str(home / "theirs"))
before = settings()
on, warned = call()
out["user_dir"] = {"on": on, "warned": warned, "unchanged": settings() == before, "settings": settings(),
                   "home_has_cache": (home / ".cache" / "pfcheeshmc").exists(), "theirs_created": (home / "theirs").exists()}

home = fresh("user_disabled")
jax.config.update("jax_enable_compilation_cache", False)
on, warned = call()
out["user_disabled"] = {"on": on, "settings": settings(), "home_has_cache": (home / ".cache" / "pfcheeshmc").exists()}

home = fresh("user_thresholds")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 100)
on, warned = call()
out["user_thresholds"] = {"on": on, "settings": settings()}

home = fresh("user_max_size")
jax.config.update("jax_compilation_cache_max_size", 123456789)
on, warned = call()
out["user_max_size"] = {"on": on, "settings": settings()}

home = fresh("unsafe_mode")
unsafe = home / ".cache" / "pfcheeshmc" / "jax"
unsafe.mkdir(parents=True)
os.chmod(unsafe, 0o777)
on, warned = call()
out["unsafe_mode"] = {"on": on, "warned": warned, "dir_setting": jax.config.jax_compilation_cache_dir}

home = fresh("owner_only_but_looser_read")     # group/other may read and enter, not write: accepted
loose = home / ".cache" / "pfcheeshmc" / "jax"
loose.mkdir(parents=True)
os.chmod(loose, 0o755)
on, warned = call()
out["readable_by_others"] = {"on": on, "warned": warned, "mode_kept": stat.S_IMODE(loose.stat().st_mode)}

home = fresh("cannot_create")
(home / ".cache").write_text("a file where the cache directory should go")
on, warned = call()
out["cannot_create"] = {"on": on, "warned": warned, "dir_setting": jax.config.jax_compilation_cache_dir}

home = fresh("xdg", XDG_CACHE_HOME=str(scratch / "xdg_cache"))
out["xdg"] = {"on": call()[0], "dir_setting": jax.config.jax_compilation_cache_dir,
              "home_has_cache": (home / ".cache" / "pfcheeshmc").exists()}
for label, value in (("relative", "relative/dir"), ("empty", "")):
    home = fresh(f"xdg_{label}", XDG_CACHE_HOME=value)
    out[f"xdg_{label}"] = {"on": call()[0], "dir_setting": jax.config.jax_compilation_cache_dir, "home": str(home)}

# JAX decides at its first compile whether the cache is in use, so the directory must work when it is set later
home = fresh("after_other_programs")
new_program()
new_program()
none_before = events["cache_misses"]
on, _ = call()
new_program()
out["after_other_programs"] = {"on": on, "misses_before": none_before, "misses_after": events["cache_misses"],
                               "entries": len(files(home / ".cache" / "pfcheeshmc" / "jax"))}

# an explicit directory, chosen after the default has been in use, is the one used from then on
home = fresh("explicit_after_default")
call()
new_program()
default = home / ".cache" / "pfcheeshmc" / "jax"
n_default = len(files(default))
pfcheeshmc.enable_compilation_cache(home / "mine")
before = settings()
new_program()
new_program()
out["explicit_after_default"] = {"entries_default_before": n_default, "entries_default_after": len(files(default)),
                                 "entries_mine": len(files(home / "mine")), "settings": before}

# the explicit helper works whatever the opt-out says
home = fresh("explicit_with_optout", **{cc.OPT_OUT_ENV: "0"})
pfcheeshmc.enable_compilation_cache(home / "mine")
new_program()
out["explicit_with_optout"] = {"entries_mine": len(files(home / "mine")), "settings": settings(),
                               "default_on": cc.ensure_default_compilation_cache()}

print(json.dumps(out))

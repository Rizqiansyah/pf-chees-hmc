from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pfcheeshmc")
except PackageNotFoundError:   # a source tree on sys.path, not installed
    __version__ = "unknown"

from .sampler import sample_pf_chees_hmc
from .compilation_cache import default_compilation_cache_dir, enable_compilation_cache

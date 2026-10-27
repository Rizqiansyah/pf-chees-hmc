"""Count XLA backend compiles, the way the compile-once tests need them counted.

JAX records one `"/jax/core/compile/backend_compile_duration"` duration event per XLA compile
(`jax/_src/interpreters/pxla.py`, `dispatch.BACKEND_COMPILE_EVENT`), tagged with `fun_name` -- the name
of the program. Listening to that event counts compiles directly; log handlers on the JAX loggers
double-count (several loggers see the same message) and miss compiles when the log level is wrong.

    with count_compiles() as cc:
        sampler.sample(...)
    assert cc.count == 0, cc.names

`count` is the number of compiles, `names` the program name of each in order, `seconds` their total
compile time. The listener is unregistered on exit, also after an exception.

`count_pytensor_functions` counts the other kind of compile: a PyTensor function, built by
`pytensor.function` / `pymc.pytensorf.compile` (rewrites, then the C backend). `FunctionMaker` is what every
such function is made by, so spying on its constructor sees them all; `sites` names the PyMC / pfcheeshmc
frames that asked for each (the innermost few), to say who compiled.

    with count_pytensor_functions() as pf:
        sampler.sample(...)
    assert pf.count == 0, pf.sites
"""
import contextlib
import dataclasses
import os
import traceback

import jax.monitoring

BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


@dataclasses.dataclass
class CompileCount:
    names: list = dataclasses.field(default_factory=list)
    seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.names)

    def __str__(self) -> str:
        return f"{self.count} compiles ({self.seconds:.2f} s): {sorted(self.names)}"


@contextlib.contextmanager
def count_compiles():
    cc = CompileCount()

    def listener(event, duration_secs, **kwargs):
        if event == BACKEND_COMPILE_EVENT:
            cc.names.append(str(kwargs.get("fun_name", "?")))
            cc.seconds += duration_secs

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield cc
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)


@dataclasses.dataclass
class PyTensorFunctionCount:
    sites: list = dataclasses.field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.sites)

    def __str__(self) -> str:
        return f"{self.count} PyTensor functions: {self.sites}"


@contextlib.contextmanager
def count_pytensor_functions():
    from pytensor.compile.function import types

    pf = PyTensorFunctionCount()
    original = types.FunctionMaker.__init__

    def spy(self, *args, **kwargs):
        frames = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in traceback.extract_stack()[:-1]
                  if "pfcheeshmc" in f.filename or os.sep + "pymc" + os.sep in f.filename]
        pf.sites.append(" > ".join(frames[-4:]))
        return original(self, *args, **kwargs)

    types.FunctionMaker.__init__ = spy
    try:
        yield pf
    finally:
        types.FunctionMaker.__init__ = original

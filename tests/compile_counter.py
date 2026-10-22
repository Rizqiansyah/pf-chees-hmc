"""Count XLA backend compiles, the way F2's compile-once tests need them counted.

JAX records one `"/jax/core/compile/backend_compile_duration"` duration event per XLA compile
(`jax/_src/interpreters/pxla.py`, `dispatch.BACKEND_COMPILE_EVENT`), tagged with `fun_name` -- the name
of the program. Listening to that event counts compiles directly; log handlers on the JAX loggers
double-count (several loggers see the same message) and miss compiles when the log level is wrong.

    with count_compiles() as cc:
        sampler.sample(...)
    assert cc.count == 0, cc.names

`count` is the number of compiles, `names` the program name of each in order, `seconds` their total
compile time. The listener is unregistered on exit, also after an exception.
"""
import contextlib
import dataclasses

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

"""PyMC graphs as JAX functions whose model inputs are runtime arguments, not compile-time constants.

``pymc.sampling.jax.get_jaxified_graph`` turns every shared variable into a constant
(``pymc/sampling/jax.py:116``) and PyTensor's JAX rewrite then folds and fuses constants, so data and
prior values end up inside the XLA program: change one and everything recompiles. `lift_graph` builds
the same function with the *model inputs* replaced by symbolic inputs first (rule C of the F2 plan):

* distribution-parameter constants (direct inputs of a model RV node, whatever their dtype), after
  widening the scalar ones (see `_widen_model_inputs`);
* observed-data constants;
* every non-scalar constant in the graph (data arrays, index arrays, constant matrices);
* every non-RNG shared variable (``pm.Data``).

Only *value* uses are lifted (see `_VALUE_OPS`); a shape, size, axis or basic-index position must stay a
compile-time constant for JAX. Formula constants (the 0.5, log 2 pi and 1 of PyMC's logp derivations)
stay static on purpose: lifting them as well blocks PyTensor's value-dependent rewrites
(``x**2 -> sqr``, ``log(1 + x) -> log1p``). The lift happens BEFORE ``mode.JAX.optimizer.rewrite``, as
afterwards the prior values have already been folded into ``Composite`` ops.

A Python scalar literal is autocast by PyTensor to the smallest dtype that holds it exactly (0, 10 ->
int8; 1000 -> int16; 0.5, 10.0 -> float32; 0.1, 10.1 -> float64), so the dtype of a prior would change
with its value, and with it the program. The scalar prior constants are therefore widened to the model's
float (or integer) dtype first, which also repairs int8 overflows such as ``pm.Uniform("u", -100, 100)``
(``upper - lower`` is NaN in PyMC 5.28.5).

The lowered StableHLO of the result then holds no data and no prior values: its sha256 is a
structure-only key, and a registry maps that key to the first function built for it so that
structurally identical models share one function object (and with it JAX's in-memory caches).

Importing this module also registers JAX's own ``erfcx`` for PyTensor's ``Erfcx`` (see `_erfcx`).
"""
from __future__ import annotations

import functools
import hashlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.special as sps
from pymc.pytensorf import constant_fold
from pymc.sampling.jax import get_jaxified_graph
from pytensor.compile import mode
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.replace import clone_replace, graph_replace
from pytensor.graph.traversal import ancestors, toposort
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.scalar.math import Erfcx
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1
from pytensor.tensor.variable import TensorConstant

__all__ = ["LiftedGraph", "lift_graph", "lift_logp"]


# --------------------------------------------------------------------------------------------------
# erfcx (F2-1)
# --------------------------------------------------------------------------------------------------
# PyTensor sends `Erfcx` to TensorFlow Probability, which has no release for this JAX. JAX has its own
# `jax.scipy.special.erfcx`, which matches SciPy to the last digit except in one band just below the
# point where it switches to its asymptotic series: there `exp(x**2) * erfc(x)` is evaluated with
# `erfc(x)` already subnormal, which XLA flushes to zero, so the result is exactly 0. The band is
# [erfcinv(finfo.tiny), sqrt(log(finfo.max))]:
#     float64  26.5432584 .. 26.6417476      float32  9.19455 .. 9.41928
# (measured against `scipy.special.erfcx` on a 1e-5 grid; erfcx there is about 0.0212 and 0.0553).
# Repro (jax 0.11.2, x64): `jax.lax.erfc(jnp.float64(26.6))` is 0.0 (SciPy: 1.09e-309), so
# `jax.scipy.special.erfcx(jnp.float64(26.6))` is 0.0 (SciPy: 0.021195).
# TODO: remove `_erfcx_guarded` and register `jax.scipy.special.erfcx` directly once JAX fixes this.

@functools.cache
def _band(dtype) -> tuple[float, float]:
    """The (low, high) edges of the band where `jax.scipy.special.erfcx` returns 0, with a margin.

    The edges are widened by 0.1% each side; the series is accurate far beyond them. Float16 and
    bfloat16 are upcast to float32 inside JAX's erfcx, so they share the float32 band.
    """
    fi = np.finfo(np.float64 if np.dtype(dtype) == np.float64 else np.float32)
    lo = float(sps.erfcinv(fi.tiny))
    hi = float(np.sqrt(np.log(fi.max)))
    return lo * (1.0 - 1e-3), hi * (1.0 + 1e-3)


# Asymptotic series erfcx(x) ~ 1/(x sqrt(pi)) * sum_k (-1)^k (2k-1)!! / (2 x^2)^k, descending powers of
# 1/x^2 for polyval. The first three terms (1 - 1/(2x^2) + 3/(4x^4)) leave a relative error of 5e-9 at
# x = 26.5, so the series is carried to nine terms, as JAX does above its own threshold (the first
# omitted term is below 1e-17 at the lower band edge in float64, 1e-8 in float32).
_SERIES_COEFFS = np.array(
    [(-1.0) ** k * float(np.prod(np.arange(1, 2 * k, 2))) / 2.0**k for k in range(8, -1, -1)]
)


def _erfcx_series(x):
    t = 1.0 / (x * x)
    sqrt_pi = jnp.asarray(np.sqrt(np.pi), dtype=x.dtype)  # a NumPy scalar would promote float32 to float64
    return jnp.polyval(jnp.asarray(_SERIES_COEFFS, dtype=x.dtype), t) / (x * sqrt_pi)


def _erfcx_guarded(x):
    """`jax.scipy.special.erfcx`, with the band where JAX returns 0 replaced by the asymptotic series.

    Safe-`where` pattern: each branch only ever sees arguments it is valid for (JAX's erfcx is fed 0
    inside the band, the series is fed the band centre outside it), so neither the values nor the
    gradients of the unselected branch can be NaN. The gradient is
    d/dx erfcx = 2 x erfcx(x) - 2/sqrt(pi) outside the band (JAX's own rule) and the derivative of the
    series inside it.
    """
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.result_type(float))
    lo, hi = _band(np.dtype(x.dtype))
    in_band = (x >= lo) & (x <= hi)
    x_jax = jnp.where(in_band, jnp.zeros_like(x), x)
    x_series = jnp.where(in_band, x, jnp.full_like(x, 0.5 * (lo + hi)))
    return jnp.where(in_band, _erfcx_series(x_series), jss.erfcx(x_jax))


# `jax_funcify` is a singledispatch: this replaces PyTensor's TFP-backed registration, and also applies
# inside `Composite` (its inner graph is funcified node by node through the same dispatch). The import
# of `pytensor.link.jax.dispatch` above has already run PyTensor's own registration, so this one wins.
@jax_funcify.register(Erfcx)
def jax_funcify_Erfcx(op, **kwargs):
    return _erfcx_guarded


# --------------------------------------------------------------------------------------------------
# lifting
# --------------------------------------------------------------------------------------------------
# A model input is lifted only if every client of it (looking through `DimShuffle`, which is how
# `ExpandDims`, `Transpose` and broadcasting are spelled) uses its *value*. Anything else (`Subtensor`
# positions, `Reshape`/`Alloc` shapes, `Join` axes, ...) keeps it static: JAX needs those at trace time.
# Two additions to the plan's list, both harmless: `Output` (the input is itself a requested output) and
# `Shape`/`Shape_i`, which read the static shape of the symbolic input, not its value. Without those a
# `pm.Data` (whose `Shape_i` PyTensor cannot fold, as it can for a constant) or an MvNormal's observed
# array (its logp calls `pt.shape(value)`) would always stay static, and its values would leak.
_VALUE_OPS = (Elemwise, CAReduce, Dot, Blockwise, AdvancedSubtensor, AdvancedSubtensor1, Output, Shape, Shape_i)

# Fallback chain, richest rule first. `rule_used` of the result is the first one that worked.
#   model_inputs              rule C, with the scalar prior constants widened (dtype independent of value)
#   model_inputs_native_dtype rule C as the plan states it: the constants keep PyTensor's autocast dtype
#   arrays_and_shared         only non-scalar constants and shared variables
#   none                      today's behaviour: every constant is compile-time
RULES = ("model_inputs", "model_inputs_native_dtype", "arrays_and_shared", "none")
_MODEL_INPUT_RULES = ("model_inputs", "model_inputs_native_dtype")


@dataclass(frozen=True)
class LiftedGraph:
    """A JAX function of ``(inputs, values)``, the current ``values``, and a structure-only key.

    fn
        ``fn(inputs, values)`` -> list of outputs, like ``jax_funcify(fgraph)`` but with the lifted
        model inputs as the second argument. Pure and JAX-traceable. Objects with equal ``key`` share
        one ``fn`` (module registry).
    values
        The current lifted values (data, prior values, shared variables) as a tuple of ``jnp`` arrays,
        in the order ``fn`` expects. Empty for ``rule_used == "none"``.
    key
        sha256 of the lowered StableHLO of ``fn`` at the abstract shapes and dtypes of ``inputs`` and
        ``values``. It holds the structure, the static constants and the shapes/dtypes, not the values.
    n_lifted
        ``len(values)``.
    sources
        The model's own variables the values were read from (constants and shared variables), in the
        order of ``values``; `current_values` re-reads the shared ones.
    kept_static
        One ``(client op name, dtype, shape)`` per candidate that was kept static because a client
        needs its value at trace time (the first such client is named).
    rule_used
        One of `RULES`: the rule that produced ``fn``, after any fallback.
    """

    fn: Callable[[Sequence[Any], Sequence[Any]], list]
    values: tuple
    key: str
    n_lifted: int
    kept_static: list
    rule_used: str
    sources: tuple = ()

    def current_values(self) -> tuple:
        """``values`` as they are now. A ``pm.set_data`` since the build is seen; constants cannot change.

        Same shapes and dtypes as ``values`` unless a shared variable was set to another shape; the
        caller compares and, if they differ, lifts again (the structure, and so the key, has changed).
        """
        return tuple(
            jnp.asarray(src.get_value(borrow=True)) if isinstance(src, SharedVariable) else value
            for src, value in zip(self.sources, self.values)
        )


# key -> the first `fn` built for it.
_REGISTRY: dict[str, Callable] = {}


def _first_bad_client(fg: FunctionGraph, var) -> str | None:
    """Name of the first client op of ``var`` that is not a value use, looking through DimShuffle."""
    for node, _ in fg.clients.get(var, []):
        if isinstance(node.op, DimShuffle):
            for out in node.outputs:
                bad = _first_bad_client(fg, out)
                if bad is not None:
                    return bad
        elif not isinstance(node.op, _VALUE_OPS):
            return type(node.op).__name__
    return None


def _widened_dtype(const_dtype, rv_dtype) -> str | None:
    """Target dtype for a scalar distribution-parameter constant, or None to leave it alone.

    A float narrower than floatX goes to floatX; an int or uint parameter goes to floatX for a
    float-valued RV and to int64 for an int-valued one. Never narrows (so uint64 and, under a float32
    floatX, int64 are left alone), and leaves bool and anything else alone.
    """
    src = np.dtype(const_dtype)
    floatx = np.dtype(pytensor.config.floatX)
    if src.kind == "f":
        target = floatx
    elif src.kind in "iu":
        target = floatx if np.dtype(rv_dtype).kind == "f" else np.dtype("int64")
    else:
        return None
    return target.name if target != src and np.promote_types(src, target) == target else None


def _constant(value, dtype):
    return pt.constant(np.asarray(value).astype(dtype), dtype=dtype)


def _fingerprint(array):
    return (array.dtype.str, array.shape, array.tobytes())


def _built_from_literals(var) -> bool:
    """True if ``var`` is computed from scalar constants alone (a literal that was broadcast or cast)."""
    roots = [v for v in ancestors([var]) if v.owner is None]
    return all(isinstance(v, TensorConstant) and v.ndim == 0 for v in roots)


def _widen_model_inputs(model, outputs):
    """``outputs`` rebuilt with the scalar distribution-parameter constants of ``model`` widened.

    Returns ``(new_outputs, replaced)`` where ``replaced`` maps each old constant to its widened one.
    `clone_replace(..., rebuild_strict=False)` remakes every downstream node through ``make_node``, so
    the new dtype propagates (``graph_replace`` would refuse the type change). The dtype rule is
    `_widened_dtype`; arrays are left alone, as NumPy arrays already carry a fixed dtype.

    Two forms of a literal reach the graph. For an RV of shape () the parameter is the scalar constant
    itself. For any RV with a non-trivial shape it is ``ExpandDims(constant)``, which PyMC's logp
    rewrites fold into a one-element constant of its own, with the literal's autocast dtype and no link
    back to the scalar. Those folded constants are found by their fingerprint (dtype, shape, bytes) and
    widened the same way. Only non-scalar ones are matched, so a formula constant such as an exponent can
    never be mistaken for a prior value.
    """
    replaced, folded = {}, {}
    for rv in model.basic_RVs:
        node = rv.owner
        params = node.op.dist_params(node) if hasattr(node.op, "dist_params") else node.inputs
        for param in params:
            if isinstance(param, TensorConstant):
                if param.ndim == 0 and param not in replaced:
                    target = _widened_dtype(param.dtype, rv.dtype)
                    if target is not None:
                        replaced[param] = _constant(param.data, target)
            elif param.owner is not None and _built_from_literals(param):
                (value,) = constant_fold([param], raise_not_constant=False)
                if isinstance(value, np.ndarray) and value.ndim > 0:
                    target = _widened_dtype(value.dtype, rv.dtype)
                    if target is not None:
                        folded.setdefault(_fingerprint(value), _constant(value, target))
    if folded:
        for node in toposort(outputs):
            for inp in node.inputs:
                if isinstance(inp, TensorConstant) and inp.ndim > 0 and inp not in replaced:
                    twin = folded.get(_fingerprint(np.asarray(inp.data)))
                    if twin is not None:
                        replaced[inp] = twin
    if not replaced:
        return list(outputs), replaced
    return clone_replace(list(outputs), replace=replaced, rebuild_strict=False), replaced


def _candidates(model, fg: FunctionGraph, outputs, rule: str, replaced=None) -> list:
    """Constants and shared variables that may be lifted, in order of first appearance in a toposort.

    The order is structural (no set or id iteration), so equal structure means equal argument order.
    ``replaced`` maps the model's own constants to their widened stand-ins in ``outputs``.
    """
    replaced = replaced or {}
    model_consts = set()  # membership only; never iterated
    if rule in _MODEL_INPUT_RULES:
        for rv in model.basic_RVs:
            model_consts.update(
                replaced.get(i, i) for i in rv.owner.inputs if isinstance(i, TensorConstant)
            )
        for rv in model.observed_RVs:
            value = model.rvs_to_values[rv]
            if isinstance(value, TensorConstant):
                model_consts.add(value)

    def wanted(var):
        if isinstance(var, SharedVariable):
            return not isinstance(var.type, RandomType)
        return isinstance(var, TensorConstant) and (var.ndim > 0 or var in model_consts)

    found, seen = [], set()
    for var in (*outputs, *(i for node in fg.toposort() for i in node.inputs)):
        if var not in seen and wanted(var):
            seen.add(var)
            found.append(var)
    return found


def _abstract(model, inputs):
    """``ShapeDtypeStruct`` per input. Unknown static dims are read off the model's initial point."""
    specs, point = [], None
    for inp in inputs:
        shape = tuple(inp.type.shape)
        if any(s is None for s in shape):
            if point is None:
                point = model.initial_point()
            shape = np.shape(point[inp.name])
        specs.append(jax.ShapeDtypeStruct(shape, jax.dtypes.canonicalize_dtype(inp.type.dtype)))
    return tuple(specs)


def _struct(array):
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


def _key(fn, input_specs, values) -> str:
    # `as_text()` omits source locations by default, so two builds of one model lower to equal text.
    def lifted(inputs, values):
        return fn(inputs, values)

    text = jax.jit(lifted).lower(input_specs, tuple(_struct(v) for v in values)).as_text()
    return hashlib.sha256(text.encode()).hexdigest()


def _build(model, inputs, outputs, rule: str):
    """One attempt with one rule: (fn, values, kept_static, sources)."""
    if rule == "none":
        # Today's behaviour: shared variables and all constants are compile-time constants.
        jitted = get_jaxified_graph(inputs=inputs, outputs=outputs)
        return (lambda xs, values: jitted(*xs)), (), [], ()

    replaced = {}
    if rule == "model_inputs":
        outputs, replaced = _widen_model_inputs(model, outputs)
    fg = FunctionGraph(outputs=outputs, clone=False)
    if any(isinstance(i, SharedVariable) and isinstance(i.type, RandomType) for i in fg.inputs):
        # Same refusal as pymc.sampling.jax._replace_shared_variables.
        raise ValueError("Graph contains shared RandomType variables which cannot be safely replaced")
    lifted, kept, kept_static = [], [], []
    for var in _candidates(model, fg, outputs, rule, replaced):
        bad = _first_bad_client(fg, var)
        if bad is None:
            lifted.append(var)
        else:
            kept.append(var)
            kept_static.append((bad, str(var.type.dtype), tuple(var.type.shape)))

    replacements = {var: var.type() for var in lifted}
    # A shared variable that stays static is still replaced by its current value, as PyMC does. The
    # direct replacement (not `_replace_shared_variables`) is what lets the others stay symbolic.
    replacements.update(
        {var: pt.constant(var.get_value(borrow=True)) for var in kept if isinstance(var, SharedVariable)}
    )
    new_outputs = graph_replace(list(outputs), replacements, strict=False)
    new_inputs = [*inputs, *(replacements[var] for var in lifted)]
    fgraph = FunctionGraph(inputs=new_inputs, outputs=new_outputs, clone=True)
    mode.JAX.optimizer.rewrite(fgraph)
    jitted = jax_funcify(fgraph)

    values = tuple(
        jnp.asarray(var.get_value(borrow=True) if isinstance(var, SharedVariable) else var.data)
        for var in lifted
    )
    return (lambda xs, values: jitted(*xs, *values)), values, kept_static, tuple(lifted)


def lift_graph(model, inputs, outputs, rule: str = "model_inputs") -> LiftedGraph:
    """Build ``fn(inputs, values)`` for ``outputs`` of ``model`` with its model inputs lifted.

    ``rule`` is the first rule to try (`RULES`). When building or lowering with a rule raises, a warning
    names the rule and the exception and the next rule is tried; the last rule ("none", today's
    constants) lets its exception through.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    inputs, outputs = list(inputs), list(outputs)
    chain = RULES[RULES.index(rule):]
    for i, name in enumerate(chain):
        try:
            fn, values, kept_static, sources = _build(model, inputs, outputs, name)
            key = _key(fn, _abstract(model, inputs), values)
        except Exception as exc:
            if i == len(chain) - 1:
                raise
            reason = (str(exc).splitlines() or [""])[0]
            warnings.warn(
                f"lift_graph: rule {name!r} failed with {type(exc).__name__}: {reason}; "
                f"falling back to {chain[i + 1]!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        return LiftedGraph(
            fn=_REGISTRY.setdefault(key, fn),
            values=values,
            key=key,
            n_lifted=len(values),
            kept_static=kept_static,
            rule_used=name,
            sources=sources,
        )


def lift_logp(model) -> LiftedGraph:
    """`lift_graph` of ``model.logp()`` over ``model.value_vars``.

    ``fn(position, values)[0]`` is the same scalar as ``get_jaxified_logp(model)(position)`` (PyMC's
    ``negative_logp=True`` default is the *positive* log density; its naming is inverted).
    """
    return lift_graph(model, list(model.value_vars), [model.logp()])

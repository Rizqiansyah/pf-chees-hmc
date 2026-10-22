"""F2a design check, rule C: lift only MODEL INPUTS (distribution-parameter constants = direct constant
inputs of RV nodes, observed-data constants, shared variables, and any array constant), keeping
formula constants of PyMC's logp derivations static so value-dependent stabilising rewrites still fire.
Lift happens BEFORE PyTensor's rewrite (so prior values cannot be folded/fused into Composite ops).
Checks per model:
  (1) logp equality vs today's get_jaxified_logp at a random point;
  (2) leakage: rebuild the logp with every lifted constant perturbed (floats x1.01+0.003, ints +1)
      and compare the lowered HLO of the two pipelines (must be identical);
  (3) rewrites retained: op histogram vs today and vs lift-everything (rule A);
  (4) CPU throughput of vmap(value_and_grad) at C=1024 vs today.
"""
import os, sys, collections, time
sys.path.insert(0, "/home/akbar/Documents/Python packages/pf-chees-hmc/tools")
import r1_bias as rb
import numpy as np
import jax, jax.numpy as jnp
import pytensor
from pytensor.graph.basic import Constant, ancestors
from pytensor.graph.replace import graph_replace
from pytensor.graph.fg import FunctionGraph
from pytensor.compile import mode
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.elemwise import Elemwise, DimShuffle, CAReduce
from pytensor.tensor.math import Dot
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1
from pytensor.tensor.variable import TensorConstant
from pytensor.scalar.math import Erfcx
import jax.scipy.special as jss
from pymc.sampling.jax import get_jaxified_logp

@jax_funcify.register(Erfcx)
def _erfcx(op, **kw):
    return jss.erfcx

VALUE_OPS = (Elemwise, CAReduce, Dot, Blockwise, AdvancedSubtensor, AdvancedSubtensor1)

def value_only(fg, v):
    for node, _ in fg.clients.get(v, []):
        if node == "output":
            continue
        if isinstance(node.op, DimShuffle):
            if not all(value_only(fg, o) for o in node.outputs):
                return False
        elif not isinstance(node.op, VALUE_OPS):
            return False
    return True

def model_input_consts(model, logp):
    cand = set()
    for rv in model.basic_RVs:
        for i in rv.owner.inputs:
            if isinstance(i, TensorConstant):
                cand.add(i)
    for rv in model.observed_RVs:
        v = model.rvs_to_values[rv]
        if isinstance(v, TensorConstant):
            cand.add(v)
    anc = [v for v in ancestors([logp]) if isinstance(v, (Constant, SharedVariable)) and not v.type.__class__.__name__.startswith("RandomGenerator")]
    shared = [v for v in anc if isinstance(v, SharedVariable)]
    arrays = [v for v in anc if isinstance(v, TensorConstant) and v.ndim > 0]
    sel = [v for v in anc if isinstance(v, TensorConstant) and (v in cand or v.ndim > 0)]
    return sel, shared

def build(model, logp, rule):
    anc = [v for v in ancestors([logp]) if isinstance(v, Constant)]
    fg0 = FunctionGraph(outputs=[logp], clone=False)
    if rule == "C":
        sel, shared = model_input_consts(model, logp)
    else:  # A: every constant
        sel, shared = anc, [v for v in ancestors([logp]) if isinstance(v, SharedVariable)]
    lift = [c for c in sel if value_only(fg0, c)]
    kept_arr = [c for c in sel if c not in lift]
    reps = {c: c.type() for c in lift}
    reps.update({s: s.type() for s in shared})
    out = graph_replace([logp], reps, strict=False)
    ins = list(model.value_vars) + list(reps.values())
    fg = FunctionGraph(inputs=ins, outputs=out, clone=True)
    mode.JAX.optimizer.rewrite(fg)
    f = jax_funcify(fg)
    vals = [jnp.asarray(np.asarray(c.data)) for c in lift] + [jnp.asarray(s.get_value()) for s in shared]
    ops = collections.Counter(type(n.op.scalar_op).__name__ if hasattr(n.op, "scalar_op") else type(n.op).__name__ for n in fg.apply_nodes)
    inner = collections.Counter()
    from pytensor.scalar.basic import Composite
    for n in fg.apply_nodes:
        so = getattr(n.op, "scalar_op", None)
        if isinstance(so, Composite):
            inner.update(type(m.op).__name__ for m in so.fgraph.apply_nodes)
    return f, lift, shared, vals, ops + inner, kept_arr

def perturb(c):
    d = np.asarray(c.data)
    if np.issubdtype(d.dtype, np.floating):
        return pytensor.tensor.constant((d * 1.01 + 0.003).astype(d.dtype), dtype=d.dtype)
    if np.issubdtype(d.dtype, np.integer) and d.ndim == 0:
        return pytensor.tensor.constant((d + 1).astype(d.dtype), dtype=d.dtype)
    return c

STAB = ("Log1p", "Log1mexp", "Softplus", "Expm1", "Erfcx", "Log1pexp", "Sqr", "Sqrt", "Pow", "IntPow")
for name in sys.argv[1:]:
    model = rb.build_model(name)
    logp = model.logp()
    ip = model.initial_point()
    rng = np.random.default_rng(0)
    x = [jnp.asarray(ip[v.name] + 0.3 * rng.standard_normal(np.shape(ip[v.name]))) for v in model.value_vars]
    f_o = get_jaxified_logp(model, negative_logp=False)
    lo = float(f_o(x))
    res = {}
    for rule in ("C", "A"):
        f, lift, shared, vals, ops, kept = build(model, logp, rule)
        lh = float(f(*x, *vals)[0])
        res[rule] = (f, lift, vals, ops)
        print(f"[{rule}] {name}: lifted={len(lift)} (+shared {len(shared)}) kept-static-candidates={len(kept)} "
              f"logp={lh:.12g} vs today {lo:.12g} rel={abs(lo-lh)/abs(lo):.1e} "
              f"stab-ops={ {k: ops[k] for k in STAB if ops.get(k)} } erfcx={'Erfcx' in ops}")
    # leakage test, rule C
    f, lift, vals, ops = res["C"]
    pert = {c: perturb(c) for c in lift}
    logp2 = graph_replace([logp], pert, strict=False)[0]
    # identify perturbed constants as the lifted set of the rebuilt graph
    newc = list(pert.values())
    anc2 = set(ancestors([logp2]))
    fg0 = FunctionGraph(outputs=[logp2], clone=False)
    lift2 = [p for p in newc if p in anc2 and value_only(fg0, p)]
    reps = {c: c.type() for c in lift2}
    out2 = graph_replace([logp2], reps, strict=False)
    fg2 = FunctionGraph(inputs=list(model.value_vars) + list(reps.values()), outputs=out2, clone=True)
    mode.JAX.optimizer.rewrite(fg2)
    f2 = jax_funcify(fg2)
    vals2 = [jnp.asarray(np.asarray(c.data)) for c in lift2]
    h1 = jax.jit(lambda *a: f(*a)[0]).lower(*x, *vals).as_text()
    h2 = jax.jit(lambda *a: f2(*a)[0]).lower(*x, *vals2).as_text()
    print(f"[C] {name}: rebuilt-with-perturbed-inputs HLO identical: {h1 == h2} (len {len(h1):,} vs today {len(jax.jit(f_o).lower(x).as_text()):,})")
    # throughput C=1024
    C = 1024
    xb = [jnp.broadcast_to(a, (C,) + a.shape) + 0.01 * jnp.arange(C).reshape((C,) + (1,) * a.ndim) / C for a in x]
    g_o = jax.jit(jax.vmap(jax.value_and_grad(lambda xs: f_o(xs))))
    g_c = jax.jit(jax.vmap(jax.value_and_grad(lambda xs, *v: f(*xs, *v)[0]), in_axes=(0,) + (None,) * len(vals)))
    def tm(fn, *a):
        jax.block_until_ready(fn(*a)); t = time.perf_counter()
        for _ in range(20):
            r = fn(*a)
        jax.block_until_ready(r); return (time.perf_counter() - t) / 20 * 1e3
    print(f"[C] {name}: vmap(value_and_grad) C=1024 CPU ms: today {tm(g_o, xb):.3f}  ruleC {tm(g_c, xb, *vals):.3f}")

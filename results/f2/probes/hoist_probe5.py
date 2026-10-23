"""F2a design check: rewrite FIRST (PyMC's own get_jaxified_graph path), THEN lift surviving constants.

Keeps value-dependent stabilising rewrites (log1p, expm1, erfcx-only-where-needed) and constant folding.
Lift rule (usage-based, whitelist): a constant is lifted iff every client (through DimShuffle) is an
Elemwise, Dot/Blockwise, CAReduce, or the data/index input of an advanced (gather) subtensor.
Reports: which client ops kept constants static, logp equality, HLO invariance to lifted values,
erfcx presence, and constant-free op counts.
"""
import os, sys, collections
MAIN = "/home/akbar/Documents/Python packages/pf-chees-hmc"
sys.path.insert(0, os.path.join(MAIN, "tools"))
import r1_bias as rb
import numpy as np
import jax, jax.numpy as jnp
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.compile import mode
from pytensor.link.jax.dispatch import jax_funcify
from pymc.sampling.jax import _replace_shared_variables, get_jaxified_graph
from pytensor.tensor.elemwise import Elemwise, DimShuffle, CAReduce
from pytensor.tensor.math import Dot
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1

VALUE_OPS = (Elemwise, CAReduce, Dot, Blockwise, AdvancedSubtensor, AdvancedSubtensor1)

for name in sys.argv[1:]:
    model = rb.build_model(name)
    logp = model.logp()
    out = _replace_shared_variables([logp])
    fg = FunctionGraph(inputs=list(model.value_vars), outputs=out, clone=True)
    mode.JAX.optimizer.rewrite(fg)
    consts = [v for v in fg.variables if isinstance(v, Constant) and fg.clients.get(v)]
    static_why = collections.Counter()
    def liftable(v):
        for node, _ in fg.clients.get(v, []):
            if node == "output":
                continue
            if isinstance(node.op, DimShuffle):
                if not all(liftable(o) for o in node.outputs):
                    return False
            elif not isinstance(node.op, VALUE_OPS):
                static_why[type(node.op).__name__] += 1
                return False
        return True
    lift = [c for c in consts if liftable(c)]
    keep = [c for c in consts if c not in lift]
    new_inputs = [c.type() for c in lift]
    fg.replace_all(list(zip(lift, new_inputs)), import_missing=True)
    for i in new_inputs:
        if i not in fg.inputs:
            fg.add_input(i)
    # reorder inputs: value vars first, then lifted
    fg2 = FunctionGraph(inputs=list(fg.inputs), outputs=list(fg.outputs), clone=False)
    f_h = jax_funcify(fg2)
    n_vv = len(model.value_vars)
    f_o = get_jaxified_graph(inputs=model.value_vars, outputs=[logp])
    ip = model.initial_point()
    rng = np.random.default_rng(0)
    vals = [jnp.asarray(ip[v.name] + 0.3 * rng.standard_normal(np.shape(ip[v.name]))) for v in model.value_vars]
    order = fg2.inputs[n_vv:]
    cvals = [jnp.asarray(np.asarray(c.data)) for c in lift]
    m = dict(zip(new_inputs, cvals))
    cv = [m[i] for i in order]
    lo = float(f_o(*vals)[0]); lh = float(f_h(*vals, *cv)[0])
    g = lambda *a: f_h(*a)[0]
    hlo1 = jax.jit(g).lower(*vals, *cv).as_text()
    cv2 = [c * 1.01 if jnp.issubdtype(c.dtype, jnp.floating) else (c + 1 if jnp.issubdtype(c.dtype, jnp.integer) and c.ndim == 0 else c) for c in cv]
    hlo2 = jax.jit(g).lower(*vals, *cv2).as_text()
    hlo_o = jax.jit(lambda *a: f_o(*a)[0]).lower(*vals).as_text()
    ops = collections.Counter(type(n.op.scalar_op).__name__ if hasattr(n.op, "scalar_op") else type(n.op).__name__ for n in fg2.apply_nodes)
    print(f"[r-then-l] {name}: consts={len(consts)} lifted={len(lift)} static={len(keep)} "
          f"static values={[np.asarray(c.data).tolist() if np.size(c.data) < 4 else f'arr{np.shape(c.data)}' for c in keep][:10]} static clients={dict(static_why)}")
    print(f"[r-then-l] {name}: lifted payload elements={sum(int(np.size(c.data)) for c in lift)}; nodes={len(fg2.apply_nodes)}; erfcx={'Erfcx' in ops}; log1p={ops.get('Log1p',0)} log1mexp={ops.get('Log1mexp',0)} softplus={ops.get('Softplus',0)}")
    print(f"[r-then-l] {name}: logp orig={lo:.12g} lifted={lh:.12g} rel={abs(lo-lh)/max(abs(lo),1e-300):.2e}; "
          f"HLO orig={len(hlo_o):,} lifted={len(hlo1):,} chars; invariant to values: {hlo1 == hlo2}")

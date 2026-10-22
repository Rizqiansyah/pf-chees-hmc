"""Fix-2 feasibility (model side): can data / prior constants become runtime inputs of the JAX logp?

For each model: list the graph constants PyMC bakes in, hoist every non-integer-scalar constant to a
symbolic input, jaxify, and check (a) logp equality vs PyMC's own jaxified graph, (b) how much
constant payload the original compiled HLO embeds vs the hoisted one.
"""
import os, sys
MAIN = "/home/akbar/Documents/Python packages/pf-chees-hmc"
sys.path.insert(0, os.path.join(MAIN, "tools"))
import r1_bias as rb
import numpy as np
import jax, jax.numpy as jnp
from pytensor.graph.basic import graph_inputs, Constant
from pytensor.graph.replace import graph_replace
from pytensor.graph.fg import FunctionGraph
from pytensor.compile import mode
from pytensor.link.jax.dispatch import jax_funcify
from pymc.sampling.jax import _replace_shared_variables, get_jaxified_graph

for name in sys.argv[1:]:
    model = rb.build_model(name)
    logp = model.logp()
    out = _replace_shared_variables([logp])
    consts = [v for v in graph_inputs(out) if isinstance(v, Constant)]
    kinds = {}
    for c in consts:
        k = (str(c.dtype), c.ndim)
        kinds[k] = kinds.get(k, 0) + int(np.asarray(c.data).size)
    MODE = os.environ.get("HOIST", "arrays")
    hoist = [c for c in consts if c.ndim >= 1] if MODE == "arrays" else [c for c in consts if not (c.ndim == 0 and "int" in str(c.dtype))]
    reps = {c: c.type() for c in hoist}
    new_out = graph_replace(out, reps, strict=False)
    fg = FunctionGraph(inputs=list(model.value_vars) + list(reps.values()), outputs=new_out, clone=True)
    mode.JAX.optimizer.rewrite(fg)
    f_h = jax_funcify(fg)
    f_o = get_jaxified_graph(inputs=model.value_vars, outputs=[logp])
    ip = model.initial_point()
    rng = np.random.default_rng(0)
    vals = [jnp.asarray(ip[v.name] + 0.3 * rng.standard_normal(np.shape(ip[v.name]))) for v in model.value_vars]
    cvals = [jnp.asarray(np.asarray(c.data)) for c in hoist]
    lo = float(f_o(*vals)[0]); lh = float(f_h(*vals, *cvals)[0])
    hlo_o = jax.jit(lambda *a: f_o(*a)[0]).lower(*vals).as_text()
    hlo_h = jax.jit(lambda *a: f_h(*a)[0]).lower(*vals, *cvals).as_text()
    # perturb every hoisted float input: same compiled program, different value
    cvals2 = [c * 1.01 if jnp.issubdtype(c.dtype, jnp.floating) else c for c in cvals]
    hlo_h2 = jax.jit(lambda *a: f_h(*a)[0]).lower(*vals, *cvals2).as_text()
    print(f"[hoist] {name}: d={sum(int(np.size(ip[v.name])) for v in model.value_vars)} "
          f"constants={len(consts)} payload(dtype,ndim->elements)={kinds} hoisted={len(hoist)}")
    print(f"[hoist] {name}: logp orig={lo:.10g} hoisted={lh:.10g} rel_err={abs(lo-lh)/max(abs(lo),1e-300):.2e}")
    print(f"[hoist] {name}: HLO text orig={len(hlo_o):,} chars, hoisted={len(hlo_h):,} chars; "
          f"hoisted HLO identical under 1% data change: {hlo_h == hlo_h2}")

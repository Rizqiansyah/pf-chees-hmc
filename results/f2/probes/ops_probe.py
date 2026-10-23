import os, sys, collections
sys.path.insert(0, "/home/akbar/Documents/Python packages/pf-chees-hmc/tools")
import r1_bias as rb
from pytensor.graph.fg import FunctionGraph
from pytensor.compile import mode
from pytensor.graph.basic import Constant
from pymc.sampling.jax import _replace_shared_variables
from pytensor.scalar.basic import Composite
for name in sys.argv[1:]:
    model = rb.build_model(name)
    fg = FunctionGraph(inputs=list(model.value_vars), outputs=_replace_shared_variables([model.logp()]), clone=True)
    mode.JAX.optimizer.rewrite(fg)
    for n in fg.toposort():
        so = getattr(n.op, "scalar_op", None)
        inner = ""
        if isinstance(so, Composite):
            ic = [c for c in so.fgraph.variables if isinstance(c, Constant)]
            inner = f" COMPOSITE inner consts={[float(c.data) for c in ic][:8]}"
        print(name, type(n.op).__name__, type(so).__name__ if so else "", [("C" if isinstance(i, Constant) else "") for i in n.inputs], inner)

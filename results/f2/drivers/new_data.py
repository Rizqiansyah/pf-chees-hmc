"""F2f helper: the same PyMC model with NEW observed data at the same shapes (any model, no special cases).

`with_new_data(model, seed)` returns a new model in which every observed variable's data constant is
permuted along axis 0 by a seeded RNG (seed 0 returns a clone with the data unchanged). Shapes, dtypes and
support are kept, so it is "the next dataset" for a compile-cost measurement: the structure is equal and
the values differ. Observed data held in a shared variable (`pm.Data`) is left alone. Built with PyMC's own
`fgraph_from_model` / `model_from_fgraph`.
"""
import numpy as np
import pytensor.tensor as pt
from pymc.model.fgraph import ModelObservedRV, fgraph_from_model, model_from_fgraph
from pytensor.tensor.variable import TensorConstant


def with_new_data(model, seed: int):
    fgraph, _ = fgraph_from_model(model)
    rng = np.random.default_rng(seed)
    for node in list(fgraph.toposort()):
        if not isinstance(node.op, ModelObservedRV):
            continue
        data = node.inputs[1]
        if not isinstance(data, TensorConstant) or data.ndim == 0 or seed == 0:
            continue
        new = rng.permutation(np.asarray(data.data), axis=0)
        fgraph.replace(data, pt.constant(new, dtype=data.dtype, name=data.name), import_missing=True)
    return model_from_fgraph(fgraph, mutate_fgraph=True)

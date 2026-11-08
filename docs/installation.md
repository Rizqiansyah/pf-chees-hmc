# Installation

Python 3.11 or later (the floor of BlackJAX 1.7). BlackJAX 1.7 or later is required: the sampler uses its internals.

## With pip

```bash
pip install .                 # CPU JAX
pip install ".[cuda13]"       # jax[cuda13]: NVIDIA driver >= 580 on Linux, a GPU of SM 7.5 or newer
```

The CUDA 13 requirements are JAX's ([JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

## With pixi

The repository ships a [pixi](https://pixi.sh) workspace that pins the GPU stack:

```bash
pixi install          # solve and create ./.pixi/envs/default
pixi run gpu-check    # prints the JAX version and the devices
```

`pixi.toml` keeps numpy below 2.5 and numba at 0.66.0 (PyTensor caps numba at 0.66, and numba 0.66 does not support numpy
2.5). It also sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX does not take most of a shared GPU's memory at import.

## Recommended environment variable on the GPU

```bash
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
```

Why: [GPU notes](gpu-notes.md#compiled-hessians-can-be-wrong).

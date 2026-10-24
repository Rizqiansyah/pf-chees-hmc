#!/bin/sh
# GPU lock: every GPU process on this shared 6 GB card runs under one flock.
# Usage: benchmarks/gpu_run.sh pixi run python benchmarks/<script>.py ...
# Sets PF_GPU_LOCK=1, which benchmarks/bias.py asserts when its device is not the CPU.
exec flock /tmp/pf-gpu.lock env PF_GPU_LOCK=1 "$@"

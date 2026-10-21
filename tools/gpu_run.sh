#!/bin/sh
# A5 GPU lock: every GPU process on this shared 6 GB card runs under one flock.
# Usage: tools/gpu_run.sh pixi run python tools/<script>.py
# Sets PF_GPU_LOCK=1 so timed cells can assert it (_common.require_gpu_lock).
exec flock /tmp/pf-gpu.lock env PF_GPU_LOCK=1 "$@"

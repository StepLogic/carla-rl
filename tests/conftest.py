"""Keep the test suite off the shared GPU.

WHY THIS EXISTS. The gated learner tests build several JAX agents in one
process. Run individually they pass; run as part of the full suite they failed
with RESOURCE_EXHAUSTED, because JAX preallocates a large fraction of device
memory on first use and every additional agent draws from what is left.

THE DEEPER REASON NOT TO USE THE GPU HERE AT ALL. This machine has one 12.3 GB
card shared by three sessions. A CARLA training run needs about 5.1 GB (the
simulator takes 2.8 GB before JAX starts) and another user's jobs take about
4.8 GB per stage. A test suite that grabs device memory can therefore fail a
training run that was already going, which is a far worse outcome than a slow
test. These tests check masking, shapes and bookkeeping on tiny networks, so
the device buys nothing.

Set JAX_PLATFORMS explicitly to override, e.g. to check the GPU path
deliberately:

    JAX_PLATFORMS=cuda PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/

Must run before JAX is first imported. conftest.py is loaded before any test
module, which is what makes this the right place for it.
"""
import os

# setdefault, never assignment: an explicit choice from the caller wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Removes the CUDA command-buffer allocation, which can fail with
# RESOURCE_EXHAUSTED even when a free-memory check has just passed -- no byte
# count predicts it. Harmless on CPU, and correct if someone overrides the
# platform back to the GPU.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

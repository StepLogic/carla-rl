"""One interpreter must host JAX-on-GPU and torch-on-GPU together."""
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU -- shadow failed"
import jax
print("jax devices:", jax.devices())
assert any("cuda" in str(d).lower() or "gpu" in str(d).lower()
           for d in jax.devices()), "jax lost the GPU in the venv"
import vae_nav.vc1_source  # noqa: F401  -- vc_models importable
print("[ig_env_check] OK")
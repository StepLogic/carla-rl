# Image-Goal Arm, Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline-ready point-goal arm trained on the Phase 1 all-negative image-goal reward: VC-1 encoder, `coll` observation key, no-recost hindsight relabelling, and the trainer/run-script wiring that runs it.

**Architecture:** Everything is additive and flag-gated onto the existing gated stack — `IMAGE_GOAL_TASK=1` (env/reward, Phase 1, committed `546c597`), `PERCEPTION=vc1` (new `LatentSource` backend), `OBS_WITH_COLLISION=1` (new obs key), `EPISODE_HER=1` + a buffer-class switch (relabel variant). The observation stays the point-goal formulation (user decision: 12-d pose block kept; goal-image latent removed from obs). Critic: γ=0.99, unclipped (default path — clipping exists only in the gated learner).

**Tech Stack:** JAX/TQC trainer (`train_carla_tqc.py`), gymnasium wrappers, `vc_models` from `dependencies/eai-vc` (VC-1 ViT-B, 768-d, ~400 MiB GPU), torch 2.5.1+cu121 in a shadow venv (system torch is cu130 and cannot use the CUDA-12.2 driver — see memory `vc1-gpu-torch-wheel`).

**Spec:** the Phase 2 design agreed in session 2026-09-18/19 (recorded in memory `image-goal-phase1-implemented`); Phase 1 code in `src/vae_nav/image_goal_reward.py` and `src/configs/train_env_config_v4_image_goal.py`.

## Global Constraints

- Work in the worktree `/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl` (repo root of this plan).
- Every pytest run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest ...` (plugin autoload dies on ROS plugins; memory rule).
- Existing gated arms are FROZEN: all changes are additive, flag-gated, and **default off = byte-identical behaviour** (house rule; e.g. the `IMAGE_GOAL_TASK` flag in the env config is written this way).
- Runs happen on the ACG cluster; local is for implementation, unit tests (no CARLA), and offline analysis. Do not launch a training run locally.
- Never edit a bash script that is running (bash reads by byte offset).
- CARLA is started from Python (`CARLA_START_SERVER=1`), never a shell launcher; kill-all-CARLA loops use `pgrep -fc 'Linux-Shippin[g]'` for counting, never pkill patterns that match your own shell.
- `info["is_success"]` from the env means "drove 200 m" — NEVER read it; the wrappers publish their own.
- The trainer's mask distinction already works: `vec_env.py:152` publishes `info["terminated"]` from the wrapper's `terminated` flag, and the trainer computes `mask = 0.0 if terminated else 1.0` — time-limit truncation bootstraps.
- Constants for the new arm come from env vars with the `IG_`/existing prefixes, exactly as Phase 1 did; the reward table itself is FROZEN (user's two revisions: no +10 success bonus, no similarity term).

---

### Task 1: The `coll` observation key (z = has_collided)

**Files:**
- Modify: `src/vae_nav/vae_goal_wrapper.py`
- Test: `tests/test_obs_coll.py` (new)

**Interfaces:**
- Consumes: `info["has_collided"]` (published by `ImageGoalReward` from Phase 1; absent → False).
- Produces: obs key `"coll"`, shape `(1,)`, float32, values in {0.0, 1.0}; flag `OBS_WITH_COLLISION` (default off); `identity()["with_coll"]`; banner mentions the key when on.

The persistent collision penalty makes the reward depend on history; conditioning on z restores Markov validity. z reaches BOTH actor and critic through the shared obs dict (jaxrl2 PixelMultiplexer projects each key separately) — the spec requires the critic; the actor seeing z is harmless because z is part of the observable state, not privileged info. NOTE for implementer: jaxrl2's PixelMultiplexer gives every non-image key its own Dense→LayerNorm→tanh projection — this is why z is its OWN key, never appended to `vector` (same reasoning as `WITH_MODE`'s comment at `vae_goal_wrapper.py:76-97`).

- [ ] **Step 1: Write the failing test**

```python
"""The coll observation key: the critic's collision flag, z = has_collided."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")


class _StubEnv:
    observation_space = {}
    action_space = None

    def reset(self, **kw):
        return np.zeros(4, np.float32), {"rgb": None, "has_collided": False}

    def step(self, a):
        return (np.zeros(4, np.float32), 0.0, False, False,
                {"rgb": None, "has_collided": True})


class _FakeSource:
    name = "stub"
    latent_dim = 4

    def latent(self, rgb):
        return np.zeros(4, np.float32)

    def identity(self):
        return {"latent_backend": "stub"}


def _make(monkeypatch, coll="1"):
    monkeypatch.setenv("OBS_WITH_COLLISION", coll)
    monkeypatch.setenv("OBS_WITH_GOAL", "0")   # keep the stub small; goal block
    # needs info fields the stub does not carry
    import vae_nav.vae_goal_wrapper as m
    import importlib
    importlib.reload(m)
    return m.VAEGoalWrapper(_StubEnv(), source=_FakeSource())


def test_coll_key_off_by_default(monkeypatch):
    w = _make(monkeypatch, coll="0")
    obs, _ = w.reset()
    assert "coll" not in obs


def test_coll_key_present_when_requested_and_carries_the_flag(monkeypatch):
    w = _make(monkeypatch, coll="1")
    obs, _ = w.reset()
    assert obs["coll"].shape == (1,) and obs["coll"].dtype == np.float32
    obs2, _r, _t, _tr, _i = w.step(np.zeros(2, np.float32))
    assert float(obs2["coll"][0]) == 1.0    # the stub steps with has_collided=True


def test_coll_defaults_to_zero_when_info_has_no_flag(monkeypatch):
    monkeypatch.setenv("OBS_WITH_COLLISION", "1")
    monkeypatch.setenv("OBS_WITH_GOAL", "0")
    import vae_nav.vae_goal_wrapper as m
    import importlib
    importlib.reload(m)
    env = _StubEnv()
    env.reset = lambda **kw: (np.zeros(4, np.float32), {"rgb": None})
    w = m.VAEGoalWrapper(env, source=_FakeSource())
    obs, _ = w.reset()
    assert float(obs["coll"][0]) == 0.0


def test_identity_records_with_coll(monkeypatch):
    w = _make(monkeypatch, coll="1")
    assert w.identity()["with_coll"] is True
```

(The `obs2` dead line above is scaffolding from the draft — delete it before committing; keep the three live assertions.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_obs_coll.py -v`
Expected: FAIL (`coll` not in obs / no `with_coll` in identity).

- [ ] **Step 3: Implement**

In `vae_goal_wrapper.py`, beside the `WITH_MODE` block:

```python
# OBS_WITH_COLLISION=1 ADDS THE CRITIC'S COLLISION FLAG z = has_collided.
# It reaches BOTH actor and critic through the shared obs dict; the policy
# seeing z is harmless (z is observable state), while the CRITIC needs it for
# Markov validity: the image-goal reward's persistent collision penalty makes
# the reward depend on history, and V(o, z) repairs that. Own key, own
# projection -- the same reasoning as WITH_MODE. Default off, so every
# existing run is byte-identical.
WITH_COLLISION = os.environ.get("OBS_WITH_COLLISION", "0") == "1"
COLL_DIM = 1
```

In `__init__` space building (after the `with_mode` block):

```python
        if self.with_collision:
            space["coll"] = spaces.Box(0.0, 1.0, (COLL_DIM,), np.float32)
```

(with `self.with_collision = WITH_COLLISION` beside `self.with_mode = WITH_MODE`, and a `", 'coll': (1,)"` term in the banner print — the banner MUST list every key; that contract is pinned by `tests/test_gated_obs.py`.)

In `_observation`, after the `with_mode` block:

```python
        if self.with_collision:
            out["coll"] = np.array(
                [float(bool(info.get("has_collided", False)))], np.float32)
```

In `identity()`:

```python
        ident["with_coll"] = bool(self.with_collision)
```

- [ ] **Step 4: Run the full obs + reward test set**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_obs_coll.py tests/test_gated_obs.py tests/test_image_goal_reward.py -q`
Expected: PASS (default-off means the pinned banner/width tests are untouched).

- [ ] **Step 5: Commit**

```bash
git add src/vae_nav/vae_goal_wrapper.py tests/test_obs_coll.py
git commit -m "Image-goal Phase 2: coll observation key (z = has_collided), flag-gated"
```

---

### Task 2: VC-1 encoder backend (`PERCEPTION=vc1`)

**Files:**
- Create: `src/vae_nav/vc1_source.py`
- Modify: `src/vae_nav/latent_source.py:137-167` (`from_env`)
- Create: `scripts/vc1_encoder_check.py`
- Test: `tests/test_vc1_source.py`

**Interfaces:**
- Consumes: `vc_models` (installed from `dependencies/eai-vc/vc_models`), torch.
- Produces: `VcSource` with the same interface as `LatentSource`: `.latent(rgb) -> (768,) float32`, `.latent_dim`, `.name == "vc1_vitb"`, `.identity() -> {"latent_backend": "vc1_vitb", "latent_dim": 768, ...}`, `.assert_matches` (inherited static method — the name it records must be what checkpoints carry).

`from_env` gains exactly one branch, keeping its "ONE place decides" contract:

```python
    if backend in ("vc1", "vc1_vitb"):
        from vae_nav.vc1_source import VcSource
        return VcSource()
    if backend not in ("vae", ""):
        ...
```

- [ ] **Step 1: Write the failing test**

```python
"""The VC-1 encoder source. Heavy tests skip unless vc_models is importable."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

vc = pytest.importorskip("vc_models")


def test_identity_names_the_backend_and_dim():
    from vae_nav.vc1_source import VcSource
    src = VcSource(device="cpu")
    assert src.name == "vc1_vitb"
    assert src.latent_dim == 768
    ident = src.identity()
    assert ident["latent_backend"] == "vc1_vitb"
    assert ident["latent_dim"] == 768


@pytest.mark.slow
def test_latent_output_shape_and_dtype():
    from vae_nav.vc1_source import VcSource
    src = VcSource(device="cpu")
    rgb = np.zeros((120, 160, 3), np.uint8)
    z = src.latent(rgb)
    assert z.shape == (768,) and z.dtype == np.float32
    assert np.isfinite(z).all()


def test_from_env_refuses_unknown_backends(monkeypatch):
    monkeypatch.setenv("PERCEPTION", "nonsense")
    from vae_nav.latent_source import from_env
    with pytest.raises(SystemExit):
        from_env()
```

If the test file needs a `slow` marker, register it in `tests/conftest.py` or drop the marker and gate on a module-level `pytestmark = pytest.mark.skipif(...)` — do NOT add a pytest plugin dependency.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_vc1_source.py -v`
Expected: FAIL (`ModuleNotFoundError: vae_nav.vc1_source`) or SKIP if vc_models is absent in the interpreter — the shape test must not silently skip forever; the implementing engineer runs the shape test once under `dependencies/vc1-venv` (which has vc_models + torch cu121) and records the result in the commit message.

- [ ] **Step 3: Implement `vc1_source.py`**

```python
"""The VC-1 visual-cortex encoder as a LatentSource backend.

Frozen Meta VC-1 ViT-B (eai-vc), 224x224 input, 768-d CLS embedding. Chosen
because VC-1's training data includes image-goal navigation; the embedding is
the goal-facing representation this arm wants. The backbone is FROZEN --
eval() and no_grad everywhere, never trained.

WHY A CLASS BESIDE LatentSource AND NOT INSIDE IT. LatentSource is the VAE
path (JAX, pickle checkpoints); MobileNetSource is the torch-features path.
VC-1 needs its own preprocessing (resize to 224, the vit_transforms normalise)
and its own failure mode -- a ~400 MB checkpoint download on first use -- so
it is its own class implementing the same small interface, registered in
latent_source.from_env, which stays the ONE place a backend is chosen.
"""
import os

import numpy as np

VC1_MODEL_NAME = "vc1_vitb"
VC1_EMBED_DIM = 768


class VcSource:
    def __init__(self, model_name=VC1_MODEL_NAME, device=None):
        self.name = model_name
        self.latent_dim = VC1_EMBED_DIM
        try:
            import torch
            from hydra import compose, initialize
            import hydra
        except ImportError as e:
            raise SystemExit(
                f"PERCEPTION=vc1 needs torch, hydra-core and vc_models "
                f"(pip install -e dependencies/eai-vc/vc_models): {e}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Compose vc_models' hydra config; compose() wraps the group under a
        # top-level "model" key, hence ["model"].
        with initialize(version_base=None):
            cfg = compose(config_name=f"model/{model_name}")
        model, embed_dim, transform, _metadata = hydra.utils.instantiate(
            cfg["model"])
        if int(embed_dim) != VC1_EMBED_DIM:
            raise SystemExit(
                f"vc_models reported embed_dim={embed_dim}, expected "
                f"{VC1_EMBED_DIM}. The observation space is built for "
                f"{VC1_EMBED_DIM}.")
        self._model = model.to(device).eval()
        self._transform = transform

        # THE ENCODER MUST NOT TRAIN. A single .train() call by anything
        # sharing the module would re-enable dropout/BN updates.
        for p in self._model.parameters():
            p.requires_grad = False

        print(f"[Latent] backend '{self.name}' (dim={self.latent_dim}, "
              f"device={device}, frozen vc_models VC-1)")

    def latent(self, rgb):
        """(H, W, 3) uint8 or float RGB -> (768,) float32."""
        import torch
        with torch.no_grad():
            x = self._prepare(rgb).to(self.device)
            z = self._model(x)
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    def _prepare(self, rgb):
        import torch
        x = torch.from_numpy(np.asarray(rgb, np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None]                  # HWC -> 1CHW
        # vc_models' vit_transforms resizes to 224 and normalises; it expects
        # a batched CHW float tensor in [0, 1].
        return self._transform(x)

    def __getattr__(self, item):
        # Lazy transform load on first use, after __init__ has run.
        if item == "_transform":
            from hydra.utils import instantiate
            import hydra
            with initialize(version_base=None):
                cfg = compose(config_name=f"model/{self.name}")
            object.__setattr__(self, "_transform",
                               instantiate(cfg["model"]["transform"]))
            return object.__getattribute__(self, item)
        raise AttributeError(item)

    def identity(self):
        return {"latent_backend": self.name, "latent_dim": self.latent_dim,
                "image_size": 224}

    @staticmethod
    def assert_matches(checkpoint_identity, source):
        return LatentSource.assert_matches(checkpoint_identity, source)
```

Implementer note: `__getattr__` lazy-loading is fiddly; if the hydra transform resolves cleanly in `__init__`, load it there and DELETE `__getattr__` — prefer the simpler constructor-time load. Verify `self._transform` on a real 120x160 frame in the probe (Step 5) before committing: if `vit_transforms` expects a PIL image instead of a tensor, pass the tensor's `[0]` as CHW→HWC uint8 `Image.fromarray` and note it in the docstring. The test in Step 4 catches whichever contract is real.

- [ ] **Step 4: Run the unit tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_vc1_source.py -q`
Expected: identity test PASS (needs vc_models importable — if the system python lacks it, run this one test with `dependencies/vc1-venv/bin/python` and note it).

- [ ] **Step 5: The probe script** (`scripts/vc1_encoder_check.py`)

A standalone check that exercises `VcSource` end-to-end: builds the source, encodes 100 random 120x160 frames, prints (device, mean latency ms/frame, peak VRAM if cuda). Run it under `dependencies/vc1-venv` (torch cu121) AND under system python (torch cu130 → expect cpu) and record both numbers in the commit message. Gate numbers for the run: GPU ≤ 10 ms/frame and ≤ 1 GiB; CPU ≤ 100 ms/frame.

- [ ] **Step 6: Commit**

```bash
git add src/vae_nav/vc1_source.py src/vae_nav/latent_source.py \
        tests/test_vc1_source.py scripts/vc1_encoder_check.py
git commit -m "Image-goal Phase 2: VC-1 ViT-B encoder backend (PERCEPTION=vc1)"
```

---

### Task 3: Run environment — GPU torch shadow venv

**Files:**
- Create: `scripts/make_ig_venv.sh`
- Create: `scripts/ig_env_check.py`

**Interfaces:**
- Produces: `dependencies/ig-venv` — a python interpreter in which BOTH the trainer's deps (JAX from system site-packages) and torch 2.5.1+cu121 (venv-local, shadowing system cu130) import, with `torch.cuda.is_available() == True` and `jax.devices()` seeing the same GPU. The run script (Task 7) activates it.

System torch is 2.13.0+cu130 and cannot use the CUDA-12.2 driver; JAX already sees the GPU. The image-goal arm needs torch-with-CUDA (VC-1 encoder) in the same process as JAX (learner). A venv with `--system-site-packages` plus a venv-local torch install shadows the system torch for that interpreter only — nothing outside the venv changes.

- [ ] **Step 1: Write the venv builder**

```bash
#!/bin/bash
# Build dependencies/ig-venv: system site-packages + torch cu121 shadow.
#
# WHY A SHADOW VENV. System torch is 2.13.0+cu130; the driver (535) exposes
# CUDA 12.2, so torch.cuda is permanently False. Downgrading the SYSTEM torch
# would touch every other consumer on this box. A --system-site-packages venv
# keeps all trainer deps from the system and installs torch cu121 into the
# venv's site-packages, which python resolves FIRST -- so this interpreter
# alone sees a working CUDA torch.
set -euo pipefail
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_VENV="$_ROOT/dependencies/ig-venv"
[ -d "$_VENV" ] && { echo "[ig-venv] already exists at $_VENV"; exit 0; }
python3 -m venv --system-site-packages "$_VENV"
"$_VENV/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
"$_VENV/bin/pip" install -e "$_ROOT/dependencies/eai-vc/vc_models"
echo "[ig-venv] built at $_VENV"
```

- [ ] **Step 2: Write the in-process check** (`scripts/ig_env_check.py`) and run it

```python
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
```

Run:
```bash
bash scripts/make_ig_venv.sh
dependencies/ig-venv/bin/python scripts/ig_env_check.py
```
Expected: torch cuda True, jax CudaDevice, import OK. If jax import breaks in the venv (system-site-packages conflicts), STOP and surface it — do not start installing JAX into the venv; that changes the learner's environment silently.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_ig_venv.sh scripts/ig_env_check.py
git commit -m "Image-goal Phase 2: GPU-torch shadow venv for VC-1 + JAX"
```

---

### Task 4: Coherence check + `make_env` wiring for `IMAGE_GOAL_TASK`

**Files:**
- Modify: `src/vae_nav/image_goal_reward.py` (add a module-level coherence function)
- Modify: `train_carla_tqc.py` (`make_env`, after the `LANE_EXPLORE` branch and BEFORE the `GATED` branch at `train_carla_tqc.py:478`; and in the `hers` construction at `train_carla_tqc.py:829-870`)
- Test: `tests/test_image_goal_coherence.py`

**Interfaces:**
- Consumes: `ImageGoalReward` (Phase 1), `IMAGE_GOAL_TASK` env flag (Phase 1, read by the env config).
- Produces: `image_goal_coherence() -> None` (raises `SystemExit` on incoherent settings); env stack `ImageGoalReward → VAEGoalWrapper → TimeLimit`.

- [ ] **Step 1: Write the failing tests**

```python
"""The image-goal arm's coherence gate: refuse configurations that cannot learn."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))


def _set(monkeypatch, **kw):
    monkeypatch.delenv("IMAGE_GOAL_TASK", raising=False)
    monkeypatch.delenv("GATED", raising=False)
    monkeypatch.delenv("OBS_WITH_GOAL", raising=False)
    monkeypatch.delenv("OBS_WITH_COLLISION", raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)
    import vae_nav.image_goal_reward as m
    import importlib
    importlib.reload(m)
    return m


def test_default_configuration_is_accepted(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="0")
    m.image_goal_coherence()          # no raise


def test_a_coherent_image_goal_run_is_accepted(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="0",
             OBS_WITH_COLLISION="1", PERCEPTION="vc1")
    m.image_goal_coherence()


def test_gating_and_image_goal_are_refused(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="1")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_the_arm_requires_the_coll_key(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", OBS_WITH_COLLISION="0")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_the_arm_requires_the_vc1_backend(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", PERCEPTION="mobilenet")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_a_shaped_or_sparse_goal_reward_is_refused(monkeypatch):
    # IMAGE_GOAL_TASK's reward is ImageGoalReward; a leftover LANE_FOLLOW=1
    # would stack a second reward below it whose output is discarded --
    # exactly the LaneFollowReward-below-GatedReward pattern, which exists
    # there because GatedReward reads two of its info keys. Nothing here
    # reads them, so refuse the dead wrapper.
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", LANE_FOLLOW="1")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_image_goal_coherence.py -q`
Expected: FAIL (`AttributeError: image_goal_coherence`).

- [ ] **Step 3: Implement the coherence function** (in `image_goal_reward.py`)

```python
IMAGE_GOAL_TASK = os.environ.get("IMAGE_GOAL_TASK", "0") == "1"


def image_goal_coherence():
    """Refuse image-goal configurations that cannot learn, BEFORE CARLA starts.

    Mirrors gated_config.check_gated_run: a run that starts and then dies at
    step 1 has still cost a server start, a world load and a GPU share.
    """
    if not IMAGE_GOAL_TASK:
        return
    if os.environ.get("GATED", "0") == "1":
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 with GATED=1 is not a supported pairing: the "
            "image-goal arm is a single-mode run. GatedReward draws a mode "
            "per episode and two critics; ImageGoalReward defines its own "
            "single reward and termination. Run one or the other.")
    if os.environ.get("OBS_WITH_COLLISION", "0") != "1":
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 requires OBS_WITH_COLLISION=1. The persistent "
            "collision penalty makes the reward depend on history; the critic "
            "must see z = has_collided or the MDP is not Markov and the "
            "critic fits noise.")
    if os.environ.get("PERCEPTION", "vae").lower() not in ("vc1", "vc1_vitb"):
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 expects PERCEPTION=vc1. The arm's checkpoint "
            "identity, latent width (768) and edge-weight calibration are "
            "tied to the VC-1 backend; running it on another encoder is an "
            "experiment change, not a run option.")
    if (os.environ.get("LANE_FOLLOW", "0") == "1"
            or os.environ.get("LANE_EXPLORE", "0") == "1"
            or os.environ.get("TIME_TO_GOAL", "0") == "1"):
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 must not stack another reward wrapper: its "
            "own reward is ImageGoalReward, and a second reward below it "
            "would be dead code the run banner would not reflect.")
```

- [ ] **Step 4: Wire `make_env`** — insert BEFORE the `GATED` branch (`train_carla_tqc.py:478`):

```python
    if os.environ.get("IMAGE_GOAL_TASK", "0") == "1":
        from vae_nav.image_goal_reward import (ImageGoalReward,
                                               image_goal_coherence)
        image_goal_coherence()
        # The Phase 1 reward owns termination (success / simulator_invalid /
        # MAX_STEPS timeout), latches the collision flag, and publishes every
        # info key the coll key and the relabel buffer read. The env's own
        # `done` is discarded inside ImageGoalReward.
        env = ImageGoalReward(env)
        env = VAEGoalWrapper(env, source=latent_source)
        if os.environ.get("NO_TIMELIMIT", "0") == "1":
            print("[Env] IMAGE_GOAL + NO_TIMELIMIT=1: no step cap (trace "
                  "collection)", flush=True)
            return env
        return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
```

- [ ] **Step 5: Run tests + the untouched-arms guard**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_image_goal_coherence.py tests/test_gated_reward.py tests/test_gated_obs.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/vae_nav/image_goal_reward.py train_carla_tqc.py \
        tests/test_image_goal_coherence.py
git commit -m "Image-goal Phase 2: coherence check + make_env wiring (IMAGE_GOAL_TASK)"
```

---

### Task 5: The relabel buffer variant (`ImageGoalRelabelBuffer`)

**Files:**
- Create: `src/vae_nav/image_goal_her.py`
- Test: `tests/test_image_goal_her.py`

**Interfaces:**
- Consumes: `EpisodeRelabelBuffer` (`src/vae_nav/episode_her.py`): `record(transition, fields, pos, hit, ...)`, `end_leg()`, `flush_episode()`, `state_fn` (= `VAEGoalWrapper.goal_block`), `_pick_goal`, `_leg_ends`, `_pos`.
- Produces: `ImageGoalRelabelBuffer(buffer, state_fn, goal_radius, n_relabel=1, min_disp=None, max_disp=inf, seed=0, max_steps=10000)` with the same public methods; `.stats()` gains `her_entry_done` (count of relabelled rows whose done fired).

THE KEY SIMPLIFICATION, restated so the implementer does not re-derive it: the Phase 1 reward is GOAL-INDEPENDENT — every component (−1, lane bonus, off-road, persistent collision) is a function of the transition alone. So a relabelled row keeps the reward it was flown with, byte for byte. Only two things change: the `"goal"` key is recomputed for the hindsight goal (via the wrapper's own `state_fn`), and `dones=1` exactly on the transition whose next state enters the hindsight goal's region. No recosting, ever — the module refuses every reward-coefficient argument by not having any.

- [ ] **Step 1: Write the failing tests**

Use the same stub-buffer pattern as `tests/test_episode_her.py` (read it first and copy its `_StubBuffer`/`_record_episode` helpers verbatim — they already know how to capture inserted rows). New behaviour to pin:

```python
"""The image-goal relabel buffer: goal-block rewrite + done-at-entry only."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.image_goal_her import ImageGoalRelabelBuffer  # noqa: E402


class _StubBuffer:
    def __init__(self):
        self.rows = []

    def insert(self, row, mode=None):
        self.rows.append(row)


def _goal_block(fields):
    # The contract shape: returns (12,) from fields, deterministic.
    return np.full(12, float(fields["goal_pos"][0]), np.float32)


def _tr(i, reward=-0.25):
    # The flown reward lives IN the transition dict -- the base class's
    # record() has no rewards kwarg; the trainer builds the transition dict
    # with `rewards` inside it (train_carla_tqc.py:1241-1245).
    return dict(observations={"latent": np.zeros(4, np.float32),
                              "goal": np.full(12, 99.0, np.float32)},
                actions=np.zeros(2, np.float32),
                rewards=np.float32(reward), masks=np.float32(1.0),
                dones=np.float32(0.0),
                next_observations={"latent": np.zeros(4, np.float32),
                                   "goal": np.full(12, 99.0, np.float32)})


def _buf(**kw):
    kw.setdefault("goal_radius", 2.0)
    kw.setdefault("min_disp", 5.0)
    return ImageGoalRelabelBuffer(_StubBuffer(), _goal_block, **kw)


def _episode(buf, xs, rewards=None):
    """One leg of positions xs (1-D), flown rewards, no terminals."""
    rewards = rewards or [-0.25] * len(xs)
    for i, x in enumerate(xs):
        buf.record(_tr(i, rewards[i]),
                   fields={"hero_pos": (x, 0.0), "hero_yaw": 0.0,
                           "goal_pos": (1e6, 0.0),
                           "action_history": np.zeros((4, 2))},
                   pos=(x, 0.0), hit=False)


def test_rewards_ride_along_unchanged():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    for row in buf.rows:
        assert float(row["rewards"]) == pytest.approx(-0.25)


def test_done_fires_only_on_the_entry_transition():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    entries = [r for r in buf.rows if float(r["dones"]) == 1.0]
    assert entries, "no hindsight row marked the goal entry"
    for row in entries:
        # The entry row's goal block was rebuilt for the hindsight goal.
        assert row["observations"]["goal"][0] != 99.0


def test_masks_mirror_dones():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    for row in buf.rows:
        assert float(row["masks"]) == (0.0 if float(row["dones"]) == 1.0
                                       else 1.0)


def test_coll_key_rides_along_untouched():
    buf = _buf()
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    for i, x in enumerate(xs):
        t = _tr(i)
        t["observations"]["coll"] = np.array([float(i % 2)], np.float32)
        buf.record(t, fields={"hero_pos": (x, 0.0), "hero_yaw": 0.0,
                              "goal_pos": (1e6, 0.0),
                              "action_history": np.zeros((4, 2))},
                   pos=(x, 0.0), hit=False)
    buf.flush_episode()
    for row in buf.rows:
        # z depends on the trajectory up to t, never on the goal: identical
        # to the flown row's value.
        assert "coll" in row["observations"]


def test_nonentry_rows_never_inherit_hit_or_lane_as_terminal():
    # Under the image-goal reward a collision or crossing ends NOTHING, so a
    # relabelled row must not become terminal because the flown step hit
    # something.
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0],
             rewards=[-0.25, -0.25, -1.75, -0.25, -0.25])  # step 2 collided
    buf.flush_episode()
    for row in buf.rows:
        if float(row["dones"]) == 1.0:
            # done rows are goal-entry rows, and their reward is the FLOWN
            # one (-1.75 or -0.25), not a recosted 0-on-arrival.
            assert float(row["rewards"]) in (-0.25, -1.75)


def test_timeout_final_frame_is_not_a_goal_source():
    # A truncated episode's last frame is a clock artefact; drawing it as a
    # hindsight goal would teach "reach wherever the episode happened to end".
    buf = _buf(n_relabel=50, max_disp=1000.0, min_disp=0.0)
    _episode(buf, list(range(0, 50, 5)))   # 10 steps
    buf.flush_episode(truncated=True)
    goals = {float(r["observations"]["goal"][0])
             for r in buf.rows if float(r["dones"]) == 1.0}
    assert 45.0 not in goals
```

(Adapt `_record`/stub details to what `EpisodeRelabelBuffer.record` actually requires — read `episode_her.py:225-262` first. `record` inserts the on-policy row immediately via `_insert`, so `buf.rows` after flush contains BOTH on-policy and relabelled rows; filter by checking `row["observations"]["goal"][0] != 99.0` for relabelled ones where needed.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_image_goal_her.py -q`
Expected: FAIL (`ModuleNotFoundError: vae_nav.image_goal_her`).

- [ ] **Step 3: Implement**

```python
"""Hindsight relabelling for the image-goal arm: goal rewrite + done-at-entry.

Subclasses EpisodeRelabelBuffer and overrides the ONLY thing that differs:
_relabeled(). The Phase 2 design's core fact is that the image-goal reward is
GOAL-INDEPENDENT -- every component is a function of the transition, not the
goal -- so a hindsight row keeps the flown reward exactly. There is no recost
path, no k_goal payout, no free-positive hazard in the reward: the arrival
test (pos[i+1] within goal_radius of the hindsight goal) sets dones=1 on the
entry transition only, and the task-relative done is exactly that entry.

Guard (a) -- min_disp >= goal_radius -- is inherited and still load-bearing:
it keeps hindsight goals far enough that reaching them took real travel.

The timeout final frame is never a goal source (flush_episode(truncated=True)
drops it): a clock artefact is not a destination. Physical terminals already
cannot appear mid-leg: success and hero destruction end the episode, and the
env_error path discards the whole episode in the trainer. z = the coll key
rides along untouched -- it depends on the trajectory up to t, never on which
goal the step is credited to.
"""
import numpy as np

from vae_nav.episode_her import EpisodeRelabelBuffer, GOAL


class ImageGoalRelabelBuffer(EpisodeRelabelBuffer):
    # The trainer's flush call site dispatches on this: base-class call sites
    # call flush_episode() bare; image-goal call sites pass truncated=.
    IMAGE_GOAL = True

    def __init__(self, buffer, state_fn, goal_radius, n_relabel=1,
                 min_disp=None, max_disp=np.inf, seed=0, max_steps=10000):
        # NO reward coefficients AT ALL: passing any would be dead weight at
        # best and a second reward function at worst. The superclass requires
        # gated=>time_to_goal only when gated=True, which is never the case
        # here.
        super().__init__(buffer, state_fn, goal_radius=goal_radius,
                         k_goal=0.0, k_collision=0.0, k_lane=0.0,
                         n_relabel=n_relabel, min_disp=min_disp,
                         max_disp=max_disp, seed=seed, max_steps=max_steps,
                         time_to_goal=False, gated=False)
        self.n_entry_done = 0

    def flush_episode(self, truncated=False):
        self._truncated = bool(truncated)
        n = super().flush_episode()
        self._truncated = False
        return n

    def _pick_goal(self, i, n, pos):
        # The timeout final frame is not a destination.
        cand_hi = n - 1 if getattr(self, "_truncated", False) else n
        cand = [t for t in range(i + 1, cand_hi)
                if self.min_disp <= float(np.linalg.norm(pos[t] - pos[i]))
                <= self.max_disp]
        if not cand:
            return None
        return int(self.rng.choice(cand))

    def _relabelled(self, i, goal, n):
        """Goal key rewritten; flown reward kept; done only at the entry."""
        row = self._clean(self._txns[i])
        obs_goal = self.state_fn({**self._fields[i], "goal_pos": goal})
        nxt_goal = self.state_fn({**self._fields[min(i + 1, n - 1)],
                                  "goal_pos": goal})
        row["observations"] = dict(row["observations"])
        row["next_observations"] = dict(row["next_observations"])
        row["observations"]["goal"] = obs_goal
        row["next_observations"]["goal"] = nxt_goal

        # TASK-RELATIVE DONE: the transition whose NEXT state enters the
        # hindsight goal's region. Everything else bootstraps -- including a
        # row that was flown with dones=1 from a truncation, because the
        # hindsight goal is reached inside the actually-flown trajectory.
        d_next = float(np.linalg.norm(
            self._pos[min(i + 1, n - 1)] - goal))
        entry = d_next < self.goal_radius
        row["dones"] = np.float32(1.0 if entry else 0.0)
        row["masks"] = np.float32(0.0 if entry else 1.0)
        if entry:
            self.n_entry_done += 1
        return row

    def stats(self):
        out = super().stats()
        out["her_entry_done"] = self.n_entry_done
        return out
```

(`record()` gains nothing; `hit`/`lane`/speed kwargs stay accepted and unused — the trainer call site stays uniform.)

- [ ] **Step 4: Run tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_image_goal_her.py tests/test_episode_her.py -q`
Expected: PASS (the parent's tests must stay green — nothing in the base class changed).

- [ ] **Step 5: Commit**

```bash
git add src/vae_nav/image_goal_her.py tests/test_image_goal_her.py
git commit -m "Image-goal Phase 2: no-recost hindsight relabelling, done-at-entry"
```

---

### Task 6: Trainer wiring — buffer selection and stats

**Files:**
- Modify: `train_carla_tqc.py:829-870` (the `hers` construction)
- Test: covered indirectly by Task 5's unit tests + a config smoke (Step 3); the branch is two lines of selection.

**Interfaces:**
- Consumes: `ImageGoalRelabelBuffer` (Task 5), `IMAGE_GOAL_TASK` + `image_goal_coherence` (Task 4).
- Produces: HER buffers of the right class for the arm in flight; the relabel buffer's `goal_radius` tied to `IG_GOAL_DISTANCE_THRESHOLD_M` (2.0).

- [ ] **Step 1: The selection code**

In the `EPISODE_HER` block, replace the single `EpisodeRelabelBuffer(...)` call with a branch:

```python
        for e in venv.envs:
            w = e
            while w is not None and not isinstance(w, VAEGoalWrapper):
                w = getattr(w, "env", None)
            if w is None:
                raise SystemExit("EPISODE_HER=1 needs VAEGoalWrapper in the stack")
            if os.environ.get("IMAGE_GOAL_TASK", "0") == "1":
                from vae_nav.image_goal_her import ImageGoalRelabelBuffer
                hers.append(ImageGoalRelabelBuffer(
                    buffer, w.goal_block,
                    goal_radius=float(os.environ.get(
                        "IG_GOAL_DISTANCE_THRESHOLD_M", 2.0)),
                    n_relabel=int(os.environ.get("EPISODE_HER_N", 1)),
                    min_disp=float(os.environ.get("HER_MIN_DISP", 5.0)),
                    seed=SEED + len(hers)))
                continue
            hers.append(EpisodeRelabelBuffer(
                # ...the EXISTING call, left byte-identical
                # (train_carla_tqc.py:837-867): buffer, w.goal_block,
                # goal_radius=GOAL_THRESHOLD, k_goal=K_GOAL, ..., time_to_goal,
                # step_cost, gated, n_relabel, min_disp, max_disp, seed...))
```

The image-goal arm never sets `time_to_goal`, `gated`, or any k_* — those kwargs do not exist on the variant.

- [ ] **Step 2: The flush call site**

Find the trainer's episode-end HER flush (grep `flush_episode` in `train_carla_tqc.py`) and make the call:

```python
                if getattr(hers[i], "IMAGE_GOAL", False):
                    hers[i].flush_episode(
                        truncated=bool(fi.get("truncated", False)))
                else:
                    hers[i].flush_episode()
```

The base class's `flush_episode()` signature is untouched — the kwarg exists only on the subclass, and the dispatch is on a class attribute rather than an isinstance import (the trainer imports nothing new).

- [ ] **Step 3: Stats line**

Add `her_entry_done` to the run's periodic stats print from `ImageGoalRelabelBuffer.stats()` (extend `stats()` in Task 5 to include `self.n_entry_done`).

- [ ] **Step 4: Verify the branch with a config smoke**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -c "import train_carla_tqc"` (import-only, no CARLA).
Expected: no import errors.

- [ ] **Step 5: Commit**

```bash
git add train_carla_tqc.py
git commit -m "Image-goal Phase 2: trainer HER selection for the image-goal arm"
```

---

### Task 7: The run script + smoke

**Files:**
- Create: `run_image_goal.sh`

**Interfaces:**
- Consumes: everything above; `dependencies/ig-venv` (Task 3).
- Produces: the launch script for the arm's training run (executed on the CLUSTER, not locally).

- [ ] **Step 1: Write the script** — modelled on `run_gated.sh`'s structure (banner dump, lock, python launch), with this env block:

```bash
export IMAGE_GOAL_TASK=1
export PERCEPTION=vc1
export OBS_WITH_COLLISION=1
export GATED=0
export EPISODE_HER=1
export DISCOUNT=0.99            # user 2026-09-19: discounted, unclipped
export TOP_QUANTILES_DROP=2     # proven regime: gamma=0.99 + drop 2 + unclipped
                                # worked in v5/v7 (tqc-drop memory: watch item only)
export TQC_KAPPA=1.0            # per-step TDs are O(1); the default derives
                                # kappa from K_GOAL and would land at 60
export IG_GOAL_DISTANCE_THRESHOLD_M=2.0
export HER_MIN_DISP=5.0         # above the 2 m payout radius: no free positives
# NO target bounds of any kind: the arm is unclipped by design (gated learner
# applies bounds per critic; the plain agent applies none).
```

Plus: `PYTHON="${PYTHON:-dependencies/ig-venv/bin/python}"`, the standard banner dump (`env | grep -E "^..."`), and the prove-alive steps per the preflight memory rules (checkpoint/label/town/seeds confirmed from the LIVE process after launch; CARLA started by Python, not shell).

- [ ] **Step 2: Local smoke (no CARLA)**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 IMAGE_GOAL_TASK=1 PERCEPTION=vc1 \
  OBS_WITH_COLLISION=1 python3 -c "import train_carla_tqc"
```
Expected: imports clean; coherence not triggered at import.

- [ ] **Step 3: Full test suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all green, including the existing gated-arm suites (they must be untouched).

- [ ] **Step 4: Commit**

```bash
git add run_image_goal.sh
git commit -m "Image-goal Phase 2: run script (gamma=0.99 unclipped, VC-1, HER no-recost)"
```

---

## Deliberately out of scope (each is its own future decision)

- **Edge weights**: `w = value_to_steps(pessimistic V, step_cost=1.0)`, no βσ — wired in the mapping harness only when this arm's first checkpoint exists (offline, local). No task here touches `value_distance.py`.
- **Offline trace collection scripts** for the buffer: the run script trains online; a dedicated collection pass is its own plan.
- **Any training run**: cluster only (memory rule); the plan ships the script, not the run.
- `COLLISION_IMPULSE_THRESHOLD` calibration (Phase 1 placeholder, unchanged).
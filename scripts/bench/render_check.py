"""Can MuJoCo render offscreen here? (plan 2026-11-10 learner validation, Task 2.)

Renders DMC cartpole swing-up at 84x84 from camera 0 for a few random steps with the backend in
MUJOCO_GL (set by the caller: egl or osmesa), saves PNGs for a visual check, and prints the frame
statistics. A blank or constant frame means the backend did not really render.
    MUJOCO_GL=egl python scripts/bench/render_check.py OUT_DIR
"""
import os
import sys
import time

import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
backend = os.environ.get("MUJOCO_GL", "?")
t0 = time.time()
from dm_control import suite  # noqa: E402  (after MUJOCO_GL is read)

env = suite.load("cartpole", "swingup", task_kwargs={"random": 0})
ts = env.reset()
rng = np.random.default_rng(0)
frames = []
for i in range(4):
    for _ in range(8):
        ts = env.step(rng.uniform(-1, 1, size=env.action_spec().shape))
    frames.append(env.physics.render(84, 84, camera_id=0))
dt = time.time() - t0
from PIL import Image  # noqa: E402

for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(out, f"{backend}_frame{i}.png"))
f = np.stack(frames)
print(f"[RenderCheck] MUJOCO_GL={backend} ok shape={f.shape} dtype={f.dtype} mean={f.mean():.1f} "
      f"std={f.std():.1f} frames_differ={bool(np.abs(f[0].astype(int) - f[-1]).mean() > 0.5)} "
      f"seconds={dt:.1f}", flush=True)

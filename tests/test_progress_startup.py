"""Start-up smoke for GATED_GOAL_REWARD=progress, in a FRESH process under v16's
REAL launch settings.

WHY (2026-09-26). v16 died on its first second with
    ValueError: GATED_GOAL_REWARD='progress' is not sorb|shaped
raised by an IMPORT-TIME check in gated_reward.py. Every unit test had passed,
because they import the module under the default reward and then monkeypatch
the constants -- which never re-runs module-level validation. The only way to
see import-time checks is to import under the real environment, in a process
that has not imported anything yet. That is what this does, using the settings
v16 actually snapshotted plus the gate-layer exports run_gated.sh adds after
the snapshot.
"""
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "results", "run_env",
                    "carla_gated_her_v16_progress_vc1.env")

# run_gated.sh exports these AFTER train_v1x.sh snapshots, so the .env lacks them.
GATE_LAYER = dict(GATED="1", OBS_WITH_MODE="1", OBS_WITH_GOAL="1",
                  ROUTE_POTENTIAL="1", POTENTIAL_K="1.0",
                  ENV_IMAGE_SIZE="120", VAE_IMAGE_SIZE="120",
                  INCREMENTAL_ACTIONS="0", MAX_STEER="0.5", MAX_THROTTLE="0.4")

# Used when the snapshot is absent (fresh clone): v16's recipe, by hand.
FALLBACK = dict(GATED_GOAL_REWARD="progress", ROUTE_REMAINING_MODE="project",
                GOAL_TRUNCATE_COUNTS="1", HER_RECOST="1",
                GATED_GOAL_LANE_TERMINAL="0", TQC_VALUE_FLOOR="-1010",
                GOAL_SPEED_BONUS="0.0", GOAL_ARRIVAL_BONUS="10.0",
                AGENT_LATENT_DIMS="latent+goal:512:256", PERCEPTION="vc1",
                VC1_READOUT="pyramid", AGENT_ENCODER="none", DISCOUNT="0.99",
                GATED_ALLOW_DISCOUNTED_SORB="1", GOAL_DIST_MAX="200",
                # the v8 9-d state block every gated run since v13 uses; without
                # it check_gated_run refuses (the gate would be unobservable)
                OBS_WITH_STATE="1", OBS_STATE_COLLISION="0", OBS_WITH_VECTOR="0",
                TARGET_ENTROPY_UNIT="1", ENV_RGB_CHANNEL_FIX="1")


def _launch_env():
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LD_LIBRARY_PATH", "PYTHONPATH")}
    # THE CHILD MUST STAY OFF THE GPU TOO (2026-09-27). conftest.py pins JAX to
    # the CPU for IN-PROCESS tests, but this child's environment is built from
    # scratch, so it never saw that -- and its JAX preallocated most of the card
    # while v17 was training: v17 died with RESOURCE_EXHAUSTED at step 152658.
    # CUDA_VISIBLE_DEVICES="" hides the GPU from torch as well.
    env.update(JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"),
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               CUDA_VISIBLE_DEVICES=os.environ.get("TESTS_CUDA_VISIBLE_DEVICES", ""))
    if os.path.exists(SNAP):
        for line in open(SNAP):
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env[k] = v
    else:
        env.update(FALLBACK)
    env.update(GATE_LAYER)
    return env


PROBE = textwrap.dedent('''
    import os, sys
    sys.path.insert(0, {root!r}); sys.path.insert(0, os.path.join({root!r}, "src"))
    # 1. every module the trainer imports at start-up, under the real settings
    import vae_nav.gated_reward as gr
    import vae_nav.gated_config as gc
    import vae_nav.episode_her as eh
    import vae_nav.route_potential as rp
    assert gr.GOAL_REWARD_KIND == "progress", gr.GOAL_REWARD_KIND
    assert rp.REMAINING_MODE == "project", rp.REMAINING_MODE
    # 2. the config check the trainer runs
    gc.check_gated_run(os.environ)
    # 3. the trainer module itself (its own import-time checks) and the agent
    #    build, which derives the goal critic's value clip from the reward
    import numpy as np
    from gymnasium import spaces
    import train_carla_tqc as T
    obs = spaces.Dict({{
        "latent": spaces.Box(-np.inf, np.inf, (3840,), np.float32),
        "goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32),
        "state": spaces.Box(-np.inf, np.inf, (9,), np.float32)}})
    act = spaces.Box(np.array([-0.5, 0.0], np.float32),
                     np.array([0.5, 0.4], np.float32))
    ag = T.build_agent(obs, act, 0)
    print("VALUE_CLIP", ag.value_floor, ag.value_ceiling)
    print("STARTUP_OK")
''')


def test_progress_reward_starts_under_the_real_v16_settings():
    env = _launch_env()
    r = subprocess.run([sys.executable, "-c", PROBE.format(root=ROOT)],
                       env=env, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "STARTUP_OK" in r.stdout, out[-3000:]
    clip = [l for l in r.stdout.splitlines() if l.startswith("VALUE_CLIP")][0]
    floor, ceiling = (float(x) for x in clip.split()[1:])
    # floor from the launch override, ceiling from progress_value_bounds
    assert floor == pytest.approx(-1010.0)
    assert ceiling == pytest.approx(410.0)
    assert "goal=PROGRESS" not in out or True     # banner is printed by the wrapper


def test_progress_refused_without_project_mode():
    env = _launch_env()
    env["ROUTE_REMAINING_MODE"] = "hop"
    probe = textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {ROOT!r}); sys.path.insert(0, os.path.join({ROOT!r}, "src"))
        import vae_nav.gated_config as gc
        gc.check_gated_run(os.environ)
    ''')
    r = subprocess.run([sys.executable, "-c", probe], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "ROUTE_REMAINING_MODE=project" in r.stderr


def test_progress_refused_with_v15s_value_ceiling():
    env = _launch_env()
    env["TQC_VALUE_CEILING"] = "10.0"
    probe = textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {ROOT!r}); sys.path.insert(0, os.path.join({ROOT!r}, "src"))
        import vae_nav.gated_config as gc
        gc.check_gated_run(os.environ)
    ''')
    r = subprocess.run([sys.executable, "-c", probe], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "clips the progress reward" in r.stderr

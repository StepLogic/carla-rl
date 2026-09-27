"""Start-up smoke for v18's recipe, in a FRESH process (the class of check that
would have caught v16's import-time crash): v13's minimum-time reward +
GOAL_LANE_GRADED + ENV_WEATHER_RANDOM + two training towns. Imports every module
the trainer loads at start-up, including the env config whose weather presets are
validated at IMPORT, and builds the agent."""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V18 = dict(GATED="1", OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_STATE="1",
           OBS_STATE_COLLISION="0", OBS_WITH_VECTOR="0", PERCEPTION="vc1",
           VC1_READOUT="pyramid", AGENT_ENCODER="none", GATED_GOAL_REWARD="sorb",
           ROUTE_REMAINING_MODE="project", ROUTE_POTENTIAL="1", POTENTIAL_K="1.0",
           GOAL_TRUNCATE_COUNTS="1", HER_RECOST="1", GOAL_ARRIVAL_BONUS="10.0",
           GOAL_SPEED_BONUS="0.75", GOAL_LANE_PENALTY="10.0",
           GOAL_OFF_ROAD_PENALTY="10.0", GOAL_COLLISION_PENALTY="10.0",
           GATED_GOAL_LANE_TERMINAL="0", TQC_VALUE_CEILING="10.0",
           TQC_VALUE_FLOOR="-3500", DISCOUNT="0.99", GATED_ALLOW_DISCOUNTED_SORB="1",
           TARGET_ENTROPY_UNIT="1", AGENT_LATENT_DIMS="latent+goal:512:256",
           STEP_COST="1.0", GATED_GOAL_MAX_STEPS="1000", GOAL_DIST_MAX="200",
           ENV_IMAGE_SIZE="120", ENV_RGB_CHANNEL_FIX="1", ENV_SKIP_IMAGE_FEATURES="1",
           GOAL_LANE_GRADED="1", GOAL_LANE_PENALTY_BROKEN="2.0",
           ENV_WEATHER_RANDOM="1", ENV_TOWNS="Town05,Town01")

PROBE = textwrap.dedent('''
    import os, sys
    sys.path.insert(0, {root!r}); sys.path.insert(0, os.path.join({root!r}, "src"))
    import vae_nav.gated_reward as gr
    assert gr.LANE_GRADED and gr.GOAL_REWARD_KIND == "sorb"
    import vae_nav.gated_config as gc
    gc.check_gated_run(os.environ)
    import configs.train_env_config_v4_image_goal as cfg
    assert cfg.WEATHER_RANDOM and len(cfg.WEATHER_PRESETS) == 14, cfg.WEATHER_PRESETS
    import vae_nav.episode_her, vae_nav.route_potential
    import numpy as np
    from gymnasium import spaces
    import train_carla_tqc as T
    obs = spaces.Dict({{
        "latent": spaces.Box(-np.inf, np.inf, (3840,), np.float32),
        "goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32),
        "state": spaces.Box(-np.inf, np.inf, (9,), np.float32)}})
    act = spaces.Box(np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.4], np.float32))
    ag = T.build_agent(obs, act, 0)
    print("VALUE_CLIP", ag.value_floor, ag.value_ceiling)
    print("STARTUP_OK")
''')


def _env(**over):
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
    env.update(V18)
    env.update(over)
    return env


def test_v18_starts():
    r = subprocess.run([sys.executable, "-c", PROBE.format(root=ROOT)], env=_env(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "STARTUP_OK" in r.stdout, (r.stdout + r.stderr)[-3000:]
    clip = [l for l in r.stdout.splitlines() if l.startswith("VALUE_CLIP")][0]
    assert [float(x) for x in clip.split()[1:]] == [-3500.0, 10.0]


def test_unknown_weather_preset_refused_at_import():
    probe = textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {ROOT!r}); sys.path.insert(0, os.path.join({ROOT!r}, "src"))
        import configs.train_env_config_v4_image_goal
    ''')
    r = subprocess.run([sys.executable, "-c", probe],
                       env=_env(ENV_WEATHER_PRESETS="ClearNoon,NotAPreset"),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "NotAPreset" in r.stderr

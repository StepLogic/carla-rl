"""USER RULE 2026-11-08: "record these plain SAC RULES SO THAT NO NEW RUN IS WRONG".

Plain SAC (with REDQ's critic ensemble), as the user set it out that day: "i want
standard SAC - logstd is learned and temperature is auto adjusted", "1) remove
value clips 2) remove actor normalization ... 3) multiply reward in environment",
"no hacks or normalisations that were not present in plain SAC or REDQ or DrQ",
and, after the review, no pre-tanh penalty, no PER, every transition equally
likely, no forced goal/explore split.

slurm/plain_sac.env holds the settings; every SAC training overlay sources it
LAST. train_carla_sac.require_plain_sac refuses to start a SAC training run that
breaks any of them, before the encoder or CARLA start. Evals, traces, recordings,
behaviour cloning and ALGO=iql are not SAC training runs and are exempt.
"""
import functools
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from train_carla_sac import plain_sac_violations, require_plain_sac  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()
PLAIN = "slurm/plain_sac.env"


@functools.lru_cache(maxsize=None)
def _resolved_items(overlays):
    # As slurm/train_sac.sbatch builds it: its own defaults, v52_reward.env, then
    # the overlay(s). A clean environment, so the caller's shell cannot leak in.
    script = ["set -a", f'REPO="{ROOT}"',
              "export TQC_VALUE_FLOOR=-1000 EXPLORE_VALUE_FLOOR=-1000",
              f'. "{ROOT}/slurm/v52_reward.env"']
    script += [f'. "{ROOT}/{o}"' for o in overlays]
    script.append("env -0")
    out = subprocess.run(["bash", "-c", "\n".join(script)], capture_output=True,
                         check=True, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                          "HOME": os.environ.get("HOME", "/tmp")})
    return tuple(kv.split("=", 1) for kv in out.stdout.decode().split("\0") if "=" in kv)


def _resolved(*overlays):
    return dict(_resolved_items(tuple(overlays)))


def _names(violations):
    return {v.split("=", 1)[0] for v in violations}


def test_the_plain_settings_file_satisfies_the_rule_over_the_job_defaults():
    assert plain_sac_violations(_resolved(PLAIN)) == []


def test_the_trainer_defaults_are_not_plain_sac():
    assert _names(plain_sac_violations({})) == {
        "TQC_VALUE_FLOOR", "TQC_VALUE_CEILING", "EXPLORE_VALUE_FLOOR",
        "EXPLORE_VALUE_CEILING", "ACTOR_Q_NORM", "IL_BC_WEIGHT0",
        "UNIFORM_TRANSITIONS", "GOAL_BATCH_FRAC"}


BREAKS = [("TQC_VALUE_FLOOR", "-1000"), ("TQC_VALUE_CEILING", "150"),
          ("EXPLORE_VALUE_FLOOR", "-200"), ("EXPLORE_VALUE_CEILING", "150"),
          ("ACTOR_Q_NORM", "1"), ("ACTOR_PRETANH_L2", "0.1"),
          ("ACTOR_BC_ALPHA", "2.5"), ("IL_BC_WEIGHT0", "1.0"),
          ("FIXED_ALPHA", "0.02"), ("ACTOR_LOG_STD", "-3.70,-0.43"),
          ("PER_ALPHA", "0.6"), ("UNIFORM_TRANSITIONS", "0"),
          ("GOAL_BATCH_FRAC", "0.5"), ("IL_EXPERT_FRAC", "0.25"),
          ("ONE_CRITIC", "0"), ("CRITIC_DROPOUT", "0.01")]


@pytest.mark.parametrize("name,value", BREAKS)
def test_each_setting_is_checked(name, value):
    env = _resolved(PLAIN)
    env[name] = value
    v = plain_sac_violations(env)
    assert len(v) == 1 and v[0].startswith(f"{name}={value}:"), v


def test_an_unreadable_value_is_a_violation_not_a_crash():
    env = _resolved(PLAIN)
    env["PER_ALPHA"] = "lots"
    assert _names(plain_sac_violations(env)) == {"PER_ALPHA"}


def test_run_q_recipe_is_refused():
    # job 464809: PER 0.6, temperature fixed at 0.02, v16's pre-tanh penalty,
    # value clips, Q / std(Q) actor loss, the old sampler and the 50/50 split
    names = _names(plain_sac_violations(_resolved("slurm/sac_scratch_2t_u5_orterm_idle_nb_a02.env")))
    assert {"PER_ALPHA", "FIXED_ALPHA", "ACTOR_PRETANH_L2", "TQC_VALUE_FLOOR",
            "ACTOR_Q_NORM", "UNIFORM_TRANSITIONS", "GOAL_BATCH_FRAC"} <= names


def test_the_local_plain_run_is_plain():
    assert plain_sac_violations(_resolved("slurm/local_c0ft_t05_sac_plain.env")) == []


def test_the_cluster_plain_run_is_plain_and_mirrors_the_local_one():
    env = _resolved("slurm/sac_c0_plain_rs10.env")
    assert plain_sac_violations(env) == []
    assert env["REWARD_SCALE"] == "10"
    assert env["ENV_TOWNS"] == "Town05,Town01"
    assert env["IL_PRELOAD_TOWNS"].split() == ["Town05", "Town01"]   # training towns only
    local = _resolved("slurm/local_c0ft_t05_sac_plain.env")
    for k in ("IL_PRELOAD_TRANSFORM", "IL_CRITIC_WARMUP_UPDATES", "IL_ONLINE_FREEZE_UPDATES",
              "IL_FREEZE_TRUNK", "HER", "DET_EPISODE_FRAC", "UPDATE_EVERY", "LEARNING_STARTS",
              "REWARD_SCALE", "TARGET_ENTROPY"):
        assert env.get(k) == local.get(k), k


def test_a_sac_training_run_refuses_to_start_and_names_every_violation():
    with pytest.raises(SystemExit) as e:
        require_plain_sac(training=True, algo="sac",
                          env={**_resolved(PLAIN), "PER_ALPHA": "0.6", "FIXED_ALPHA": "0.02"})
    msg = str(e.value)
    assert "PER_ALPHA=0.6" in msg and "FIXED_ALPHA=0.02" in msg
    assert "slurm/plain_sac.env" in msg
    require_plain_sac(training=True, algo="sac", env=_resolved(PLAIN))      # plain: starts


def test_evals_recordings_and_iql_are_exempt():
    require_plain_sac(training=False, algo="sac", env={"PER_ALPHA": "0.6"})
    require_plain_sac(training=True, algo="iql", env={"PER_ALPHA": "0.6"})


def test_main_checks_the_rule_before_the_encoder_and_carla_start():
    main_src = SRC[SRC.index("\ndef main():"):]
    i = main_src.index("require_plain_sac(")
    assert main_src.index("run_bc_train(OUT") < i            # cloning is not SAC
    assert i < main_src.index("DebitTinyEncoder(") and i < main_src.index("CarlaServer(")
    assert "training=not (EVAL_FROM or EXPERT_RECORD)" in main_src[i:i + 200]

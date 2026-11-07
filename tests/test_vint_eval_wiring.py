"""EVAL_POLICY=vint: run_eval drives ViNT (torch_rl.vint_driver) on c0's exact
episodes (user 2026-11-07: "evaluate" ViNT like c0). The episodes, the faults
and the scoring are run_eval's; only the action source changes."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_eval_policy_is_sac_or_vint():
    from train_carla_sac import validate_eval_policy
    assert validate_eval_policy("sac") == "sac" and validate_eval_policy("vint") == "vint"
    with pytest.raises(ValueError):
        validate_eval_policy("nomad")


def test_pixel_frames_are_what_the_recorder_stored():
    """run_record kept pxr.to_frame(info["debit_cur"], MONO_IMG) as uint8; ViNT
    learned from those, so the eval hands it the same thing, goal included."""
    from torch_rl import pixel_rows as pxr
    from train_carla_sac import pixel_frames
    info = {"debit_cur": np.full((64, 64, 3), 7, np.uint8),
            "debit_goal": np.full((256, 256, 3), 9, np.uint8)}
    cur, goal = pixel_frames(info, pxr)
    assert cur.shape == goal.shape == (pxr.MONO_IMG, pxr.MONO_IMG, 3)
    assert cur.dtype == goal.dtype == np.uint8
    assert int(cur[0, 0, 0]) == 7 and int(goal[0, 0, 0]) == 9
    assert 'pxr.to_frame(info["debit_cur"], pxr.MONO_IMG).astype(np.uint8)' in SRC


def test_main_builds_the_vint_driver_before_any_sac_agent():
    main_src = SRC[SRC.index("\ndef main():"):]
    i_vint = main_src.index('if EVAL_POLICY == "vint":')
    assert i_vint < main_src.index("agent = build_agent(")
    assert "load_vint_policy(EVAL_FROM" in main_src[i_vint:]
    # ViNT's distance is in steps of stride frames, 0.1 s each
    assert 'step_s=meta["stride"] * meta["frame_dt_s"]' in main_src[i_vint:]


def test_run_eval_takes_the_pixel_path_for_a_pixel_policy():
    body = SRC[SRC.index("\ndef run_eval("):SRC.index("\ndef coverage_perturbation(")]
    assert 'getattr(agent, "drives_from_pixels", False)' in body
    assert "agent.act_env(" in body and "pixel_frames(info, pxr)" in body
